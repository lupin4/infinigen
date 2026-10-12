# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Karhan Kayan, Alexander Raistrick

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import bpy
import fcl
import numpy as np
import procfunc as pf
import trimesh

from infinigen2.scenes.placement import forgeo

__all__ = [
    "CollisionSet",
    "any_self_collision",
    "box_intersection_test",
    "collision_set",
    "intersection_test",
    "n_colliders",
    "raycast",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollisionSet:
    objs: list[pf.Object]
    collision_manager: trimesh.collision.CollisionManager | None  # None with forGeo
    object_names: dict[int, str]
    mesh_colliders: dict[int, trimesh.Trimesh]
    mesh_fcl_colliders: dict[int, Any]
    object_fcl_objs: dict[int, Any]  # object_id -> fcl.CollisionObject in manager
    object_last_transforms: dict[int, np.ndarray]  # object_id -> last synced transform
    # forGeo backend (forgeo.LIB set): BVH per mesh, and per object its BVH and
    # world bounds as of object_last_transforms
    mesh_bvhs: dict[int, forgeo.MeshBVH]
    object_bvhs: dict[int, forgeo.MeshBVH]
    object_bounds: dict[int, np.ndarray]


def _is_collidable_object(obj: pf.Object) -> bool:
    item = obj.item()
    data = getattr(item, "data", None)
    if data is None:
        return False
    if not hasattr(data, "vertices") or not hasattr(data, "polygons"):
        return False
    return True


def _mesh_name(obj: pf.Object) -> str:
    return obj.item().name


def _mesh_id(obj: pf.Object) -> int:
    return id(obj.item().data)


def _object_id(obj: pf.Object) -> int:
    return id(obj.item())


def _mesh_from_object(obj: pf.Object) -> trimesh.Trimesh:
    vertices = pf.ops.attr.vertex_positions(obj, global_coords=False)
    if vertices.shape[0] == 0:
        raise ValueError(f"{obj.item().name} has no vertices")
    obj.item().data.calc_loop_triangles()
    loop_triangles = obj.item().data.loop_triangles
    if len(loop_triangles) == 0:
        raise ValueError(f"{obj.item().name} has no loop triangles")
    faces_flat = np.zeros(len(loop_triangles) * 3, dtype=np.int32)
    loop_triangles.foreach_get("vertices", faces_flat)
    faces = faces_flat.reshape(-1, 3)

    # Filter degenerate (zero-area) faces
    areas = np.linalg.norm(
        np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        ),
        axis=1,
    )
    valid = areas > 1e-10
    n_removed = len(faces) - int(valid.sum())
    if n_removed > 0:
        faces = faces[valid]

    dims = vertices.max(axis=0) - vertices.min(axis=0)
    logger.debug(
        "collider %s: %d verts, %d faces (%d degenerate removed), "
        "dims=(%.2f, %.2f, %.2f)",
        obj.item().name,
        len(vertices),
        len(faces),
        n_removed,
        *dims,
    )

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _object_transform(obj: pf.Object) -> np.ndarray:
    return np.asarray(obj.item().matrix_world, dtype=np.float64)


def _assert_rigid(obj: pf.Object, T: np.ndarray, tol: float = 1e-5) -> None:
    R = T[:3, :3]
    err = np.max(np.abs(R.T @ R - np.eye(3)))
    if err > tol:
        raise ValueError(
            f"object {obj.item().name!r} has non-unit scale/shear in "
            f"matrix_world (|R^T R - I|_inf = {err:.3g}); apply scale to "
            f"mesh data (e.g. `bpy.ops.object.transform_apply`) before "
            f"adding to CollisionSet"
        )


def _fcl_transform(transform: np.ndarray) -> fcl.Transform:
    return fcl.Transform(transform[:3, :3], transform[:3, 3])


def _register_object(
    manager: trimesh.collision.CollisionManager,
    name: str,
    col_obj: fcl.CollisionObject,
    fcl_geom: Any,
) -> None:
    if name in manager._objs:
        manager._manager.unregisterObject(manager._objs[name]["obj"])
    manager._objs[name] = {"obj": col_obj, "geom": fcl_geom}
    manager._names[id(fcl_geom)] = name
    manager._manager.registerObject(col_obj)


def _add_object_cached(
    manager: trimesh.collision.CollisionManager,
    name: str,
    col_obj: fcl.CollisionObject,
    fcl_geom: Any,
) -> None:
    _register_object(manager, name, col_obj, fcl_geom)
    manager._manager.update()


def _sync_transforms(col: CollisionSet) -> None:
    """Update FCL collision objects whose Blender transforms have changed since the
    last query. Callers refresh the view layer first."""
    any_changed = False
    for obj in col.objs:
        oid = _object_id(obj)
        col_obj = col.object_fcl_objs.get(oid)
        if col_obj is None:
            continue
        T = np.asarray(obj.item().matrix_world, dtype=np.float64)
        last_T = col.object_last_transforms.get(oid)
        if last_T is None or not np.array_equal(last_T, T):
            _assert_rigid(obj, T)
            col_obj.setTransform(_fcl_transform(T))
            col.object_last_transforms[oid] = T
            any_changed = True
    if any_changed:
        col.collision_manager._manager.update()


def _mesh_bvh(
    mesh_bvhs: dict[int, forgeo.MeshBVH], mesh_id: int, mesh: trimesh.Trimesh
) -> forgeo.MeshBVH:
    bvh = mesh_bvhs.get(mesh_id)
    if bvh is None:
        # convexity decides solid vs hollow, exactly as trimesh's _get_fcl_obj
        # picks fcl.Convex vs fcl.BVHModel
        bvh = forgeo.build(mesh.vertices, mesh.faces, mesh.is_convex)
        mesh_bvhs[mesh_id] = bvh
    return bvh


def _forgeo_colliders(col: CollisionSet):
    """(BVH, transform, world bounds) per collider. Like _sync_transforms, a
    moved collider is re-checked for rigidity and its bounds refreshed."""
    for obj in col.objs:
        oid = _object_id(obj)
        T = _object_transform(obj)
        if not np.array_equal(col.object_last_transforms.get(oid), T):
            _assert_rigid(obj, T)
            col.object_last_transforms[oid] = T
            col.object_bounds[oid] = forgeo.world_bounds(col.object_bvhs[oid], T)
        yield col.object_bvhs[oid], T, col.object_bounds[oid]


def _bounds_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a[0] <= b[1]) and np.all(b[0] <= a[1]))


def _forgeo_any_overlap(col: CollisionSet, probe: Any, T: np.ndarray) -> bool:
    bounds = forgeo.world_bounds(probe, T)
    for bvh, T_c, bounds_c in _forgeo_colliders(col):
        if _bounds_overlap(bounds, bounds_c) and forgeo.overlap(bvh, T_c, probe, T):
            return True
    return False


def n_colliders(col: CollisionSet) -> int:
    return len(col.object_names)


def collision_set(
    objs: Sequence[pf.Object], cache: CollisionSet | None = None
) -> CollisionSet:
    mesh_colliders: dict[int, trimesh.Trimesh] = (
        {} if cache is None else dict(cache.mesh_colliders)
    )
    mesh_fcl_colliders: dict[int, Any] = (
        {} if cache is None else dict(cache.mesh_fcl_colliders)
    )
    object_names: dict[int, str] = {}
    object_fcl_objs: dict[int, Any] = {}
    object_last_transforms: dict[int, np.ndarray] = {}
    mesh_bvhs: dict[int, forgeo.MeshBVH] = (
        {} if cache is None else dict(cache.mesh_bvhs)
    )
    object_bvhs: dict[int, forgeo.MeshBVH] = {}
    object_bounds: dict[int, np.ndarray] = {}
    manager = None if forgeo.LIB is not None else trimesh.collision.CollisionManager()
    collidable_objs: list[pf.Object] = []

    bpy.context.view_layer.update()
    for obj in objs:
        if not _is_collidable_object(obj):
            continue
        T = _object_transform(obj)
        _assert_rigid(obj, T)
        collidable_objs.append(obj)
        mesh_id = _mesh_id(obj)
        if mesh_id not in mesh_colliders:
            try:
                mesh_colliders[mesh_id] = _mesh_from_object(obj)
            except ValueError:
                collidable_objs.pop()
                continue

        if forgeo.LIB is not None:
            oid = _object_id(obj)
            bvh = _mesh_bvh(mesh_bvhs, mesh_id, mesh_colliders[mesh_id])
            object_names[oid] = _mesh_name(obj)
            object_bvhs[oid] = bvh
            object_last_transforms[oid] = T
            object_bounds[oid] = forgeo.world_bounds(bvh, T)
            continue

        fcl_geom = mesh_fcl_colliders.get(mesh_id)
        if fcl_geom is None:
            fcl_geom = manager._get_fcl_obj(mesh_colliders[mesh_id])
            mesh_fcl_colliders[mesh_id] = fcl_geom

        name = _mesh_name(obj)
        col_obj = fcl.CollisionObject(fcl_geom, _fcl_transform(T))
        _register_object(manager, name, col_obj, fcl_geom)
        oid = _object_id(obj)
        object_names[oid] = name
        object_fcl_objs[oid] = col_obj
        object_last_transforms[oid] = T
    if manager is not None:
        manager._manager.update()

    return CollisionSet(
        objs=collidable_objs,
        collision_manager=manager,
        object_names=object_names,
        mesh_colliders=mesh_colliders,
        mesh_fcl_colliders=mesh_fcl_colliders,
        object_fcl_objs=object_fcl_objs,
        object_last_transforms=object_last_transforms,
        mesh_bvhs=mesh_bvhs,
        object_bvhs=object_bvhs,
        object_bounds=object_bounds,
    )


def intersection_test(col: CollisionSet, obj: pf.Object) -> bool:
    if n_colliders(col) == 0:
        return False
    if not _is_collidable_object(obj):
        return False
    bpy.context.view_layer.update()
    T = _object_transform(obj)
    _assert_rigid(obj, T)
    if forgeo.LIB is None:
        _sync_transforms(col)

    mesh_id = _mesh_id(obj)
    geom = col.mesh_colliders.get(mesh_id)
    if geom is None:
        try:
            geom = _mesh_from_object(obj)
        except ValueError:
            return False
        col.mesh_colliders[mesh_id] = geom

    if forgeo.LIB is not None:
        return _forgeo_any_overlap(col, _mesh_bvh(col.mesh_bvhs, mesh_id, geom), T)

    fcl_geom = col.mesh_fcl_colliders.get(mesh_id)
    if fcl_geom is None:
        fcl_geom = col.collision_manager._get_fcl_obj(geom)
        col.mesh_fcl_colliders[mesh_id] = fcl_geom

    probe = trimesh.collision.CollisionManager()
    _add_object_cached(
        probe,
        "__probe__",
        fcl.CollisionObject(fcl_geom, _fcl_transform(T)),
        fcl_geom,
    )
    hit = col.collision_manager.in_collision_other(probe, return_data=False)
    return bool(hit)


def box_intersection_test(
    col: CollisionSet,
    transform: np.ndarray,
    size: float | tuple[float, float, float] = 1.0,
) -> bool:
    if n_colliders(col) == 0:
        return False
    bpy.context.view_layer.update()
    extents = np.broadcast_to(np.asarray(size, dtype=float), 3)
    if forgeo.LIB is not None:
        box = trimesh.creation.box(extents=extents)  # convex, so solid like fcl.Box
        return _forgeo_any_overlap(col, _mesh_bvh({}, 0, box), transform)
    _sync_transforms(col)
    fcl_geom = fcl.Box(*extents)
    probe = trimesh.collision.CollisionManager()
    _add_object_cached(
        probe,
        "__probe__",
        fcl.CollisionObject(fcl_geom, _fcl_transform(transform)),
        fcl_geom,
    )
    return bool(col.collision_manager.in_collision_other(probe, return_data=False))


def any_self_collision(col: CollisionSet) -> bool:
    """Return True if any two objects in the set currently intersect each other."""
    if n_colliders(col) < 2:
        return False
    bpy.context.view_layer.update()
    if forgeo.LIB is not None:
        colliders = list(_forgeo_colliders(col))
        for i, (bvh_a, T_a, bounds_a) in enumerate(colliders):
            for bvh_b, T_b, bounds_b in colliders[i + 1 :]:
                if _bounds_overlap(bounds_a, bounds_b) and forgeo.overlap(
                    bvh_a, T_a, bvh_b, T_b
                ):
                    return True
        return False
    _sync_transforms(col)
    return bool(col.collision_manager.in_collision_internal())


def raycast(
    col: CollisionSet, ray_origins: np.ndarray, ray_directions: np.ndarray
) -> Any:
    """Raycast over the combined mesh from tracked colliders."""
    if n_colliders(col) == 0:
        return (
            np.empty((0, 3)),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    bpy.context.view_layer.update()
    if forgeo.LIB is not None:
        return _forgeo_raycast(col, ray_origins, ray_directions)
    meshes = []
    for obj in col.objs:
        mesh = col.mesh_colliders.get(_mesh_id(obj))
        if mesh is None:
            continue
        mesh = mesh.copy()
        mesh.apply_transform(_object_transform(obj))
        meshes.append(mesh)

    if not meshes:
        return (
            np.empty((0, 3)),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    combined = trimesh.util.concatenate(meshes)
    return combined.ray.intersects_location(
        ray_origins=np.asarray(ray_origins),
        ray_directions=np.asarray(ray_directions),
        multiple_hits=False,
    )


def _forgeo_raycast(col: CollisionSet, ray_origins, ray_directions):
    """Nearest hit per ray over every collider, shaped like trimesh's
    intersects_location: index_tri counts faces in col.objs order, as in the
    concatenated mesh the trimesh path builds."""
    origins = np.asarray(ray_origins, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(ray_directions, dtype=np.float64).reshape(-1, 3)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    best_t = np.full(len(origins), np.inf)
    best_tri = np.full(len(origins), -1, dtype=np.int64)
    offset = 0
    for bvh, T, _bounds in _forgeo_colliders(col):
        t, face = forgeo.raycast(bvh, T, origins, directions)
        closer = (face >= 0) & (t < best_t)
        best_t[closer] = t[closer]
        best_tri[closer] = face[closer] + offset
        offset += len(bvh.faces)
    index_ray = np.flatnonzero(best_tri >= 0)
    locations = origins[index_ray] + best_t[index_ray, None] * directions[index_ray]
    return locations, index_ray, best_tri[index_ray]
