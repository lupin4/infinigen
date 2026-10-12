# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""forGeo BVH queries for placement collision, bound through ctypes.

forGeo is a separate Zig/Fortran geometry library; this module binds its C ABI
and nothing else. It is opt-in: set ``INFINIGEN_FORGEO_LIB`` to the path of
``libforgeo.dylib`` and ``collision`` routes its queries here instead of
python-fcl / trimesh. Unset, ``LIB`` is None and nothing changes.

forGeo conventions: vertices interleaved (float32 for overlap, float64 for
rays), faces 1-based int32, BVH node bounds float64. ``overlap`` follows fcl as trimesh drives it: surfaces must
cross, except that a convex mesh is solid.
"""

from __future__ import annotations

import ctypes
import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_ENV = "INFINIGEN_FORGEO_LIB"


def _arr(dtype) -> type:
    return np.ctypeslib.ndpointer(dtype=dtype, flags="C_CONTIGUOUS")


_F32, _F64, _I32, _U32 = (
    _arr(np.float32),
    _arr(np.float64),
    _arr(np.int32),
    _arr(np.uint32),
)


def _load() -> ctypes.CDLL | None:
    path = os.environ.get(_ENV)
    if not path:
        return None
    # an explicit path that does not load is a configuration error, not a reason
    # to fall back silently to the slow path
    lib = ctypes.CDLL(path)
    lib.fgeo_bvh_node_capacity.argtypes = [ctypes.c_int32]
    lib.fgeo_bvh_node_capacity.restype = ctypes.c_int32
    lib.fgeo_bvh_build_f64.argtypes = [
        ctypes.c_int32,
        _F64,
        _F64,
        _U32,
        _F64,
        _F64,
        _I32,
    ]
    lib.fgeo_bvh_build_f64.restype = ctypes.c_int32
    lib.fgeo_mesh_overlap_stack_len.argtypes = []
    lib.fgeo_mesh_overlap_stack_len.restype = ctypes.c_int32
    mesh = [
        _F32,
        ctypes.c_int32,
        _I32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        _F64,
        _F64,
        _I32,
    ]
    lib.fgeo_mesh_overlap_f32.argtypes = [
        *mesh,
        *mesh,
        _F64,
        _I32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.fgeo_mesh_overlap_f32.restype = ctypes.c_int32
    lib.fgeo_raycast_batch_f64.argtypes = [
        *[_F64, _I32, ctypes.c_int32],  # mesh
        *[_F64, _F64, ctypes.c_int32, ctypes.c_int32, ctypes.c_double],  # rays
        *[ctypes.c_int32, _F64, _F64, _I32, _I32],  # tree + stack
        *[_F64, _I32],  # out_t, out_face
    ]
    lib.fgeo_raycast_batch_f64.restype = ctypes.c_int32
    logger.info(f"forGeo collision backend: {path}")
    return lib


LIB = _load()
# one traversal stack, sized for the pair descent (it also covers single-tree
# ray traversal, which needs 64); placement is single-threaded
_STACK = None if LIB is None else np.zeros(LIB.fgeo_mesh_overlap_stack_len(), np.int32)


@dataclass(frozen=True)
class MeshBVH:
    verts: np.ndarray  # (V, 3) float32, for overlap
    verts64: np.ndarray  # (V, 3) float64, for raycast
    faces: np.ndarray  # (F, 3) int32, 1-based
    convex: bool
    node_lo: np.ndarray  # (N, 3) float64
    node_hi: np.ndarray  # (N, 3) float64
    node_child: np.ndarray  # (N, 2) int32

    @property
    def n_nodes(self) -> int:
        return len(self.node_child)


def build(vertices: np.ndarray, faces: np.ndarray, convex: bool) -> MeshBVH:
    """BVH over a mesh's triangles; ``faces`` are 0-based as trimesh holds them."""
    verts = np.ascontiguousarray(vertices, dtype=np.float32)
    verts64 = np.ascontiguousarray(vertices, dtype=np.float64)
    tri = verts64[faces]
    faces1 = np.ascontiguousarray(faces, dtype=np.int32) + 1
    lo = np.ascontiguousarray(tri.min(axis=1))
    hi = np.ascontiguousarray(tri.max(axis=1))
    n = len(faces1)
    cap = LIB.fgeo_bvh_node_capacity(n)
    node_lo = np.zeros((cap, 3))
    node_hi = np.zeros((cap, 3))
    node_child = np.zeros((cap, 2), np.int32)
    built = LIB.fgeo_bvh_build_f64(
        n, lo, hi, np.zeros(n, np.uint32), node_lo, node_hi, node_child
    )
    if built != cap:
        raise RuntimeError(f"fgeo_bvh_build_f64 returned {built}, expected {cap} nodes")
    return MeshBVH(verts, verts64, faces1, convex, node_lo, node_hi, node_child)


_UNIT_CORNERS = np.array([[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)], float)


def world_bounds(m: MeshBVH, transform: np.ndarray) -> np.ndarray:
    """(2, 3) world AABB of the root node's box under ``transform``."""
    lo, hi = m.node_lo[0], m.node_hi[0]
    corners = lo + _UNIT_CORNERS * (hi - lo)
    world = corners @ transform[:3, :3].T + transform[:3, 3]
    return np.stack([world.min(axis=0), world.max(axis=0)])


def overlap(
    a: MeshBVH, a_transform: np.ndarray, b: MeshBVH, b_transform: np.ndarray
) -> bool:
    b_to_a = np.ascontiguousarray((np.linalg.inv(a_transform) @ b_transform)[:3])
    fa, fb = ctypes.c_int32(), ctypes.c_int32()
    r = LIB.fgeo_mesh_overlap_f32(
        a.verts,
        len(a.verts),
        a.faces,
        len(a.faces),
        int(a.convex),
        a.n_nodes,
        a.node_lo,
        a.node_hi,
        a.node_child,
        b.verts,
        len(b.verts),
        b.faces,
        len(b.faces),
        int(b.convex),
        b.n_nodes,
        b.node_lo,
        b.node_hi,
        b.node_child,
        b_to_a,
        _STACK,
        ctypes.byref(fa),
        ctypes.byref(fb),
    )
    if r < 0:
        raise RuntimeError(f"fgeo_mesh_overlap_f32 returned {r}")
    return r == 1


# trimesh's ray test keeps a hit up to 1e-6 BEHIND the origin, so a ray that
# starts on a surface (an object flush on a wall) hits it. forGeo's f64 rays keep
# t > 1e-12; starting them 1e-6 + 1e-12 back gives trimesh's window. f64 because
# that contact is decided at the 1e-7 scale where f32 local coordinates round.
# (trimesh's rtree broadphase additionally drops a hit behind the origin when the
# triangle's box is too thin to reach its padded ray box -- an axis-aligned wall;
# that quirk is deliberately not reproduced.)
_START_BEHIND = 1e-6 + 1e-12


def raycast(
    m: MeshBVH, transform: np.ndarray, origins: np.ndarray, unit_dirs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest hit per world-space ray: (distance along the unit direction from
    the origin, face), face -1 on a miss."""
    inv = np.linalg.inv(transform)
    start = origins - _START_BEHIND * unit_dirs
    o = np.ascontiguousarray(start @ inv[:3, :3].T + inv[:3, 3])
    d = np.ascontiguousarray(unit_dirs @ inv[:3, :3].T)
    n = len(o)
    t = np.zeros(n)
    face = np.zeros(n, np.int32)
    LIB.fgeo_raycast_batch_f64(
        m.verts64,
        m.faces,
        len(m.faces),
        o,
        d,
        n,
        0,
        np.inf,
        m.n_nodes,
        m.node_lo,
        m.node_hi,
        m.node_child,
        _STACK,
        t,
        face,
    )
    return t - _START_BEHIND, face
