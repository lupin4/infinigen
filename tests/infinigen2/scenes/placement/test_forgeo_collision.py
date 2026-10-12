# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""The forGeo backend must answer every collision query exactly as fcl/trimesh."""

import numpy as np
import procfunc as pf
import pytest
from scipy.spatial.transform import Rotation

from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement import forgeo

pytestmark = pytest.mark.skipif(
    forgeo.LIB is None, reason="set INFINIGEN_FORGEO_LIB to test the forGeo backend"
)

_SHAPES = [
    pf.ops.primitives.mesh_cube,  # convex: solid in both backends
    pf.ops.primitives.mesh_icosphere,
    pf.ops.primitives.mesh_torus,  # non-convex: hollow
    pf.ops.primitives.mesh_monkey,  # non-convex and open
]


def _scatter(rng: np.random.Generator, n: int, extent: float) -> list:
    objs = []
    for i in range(n):
        obj = _SHAPES[i % len(_SHAPES)]()
        obj.item().location = rng.uniform(-extent, extent, 3)
        obj.item().rotation_euler = rng.uniform(0, 2 * np.pi, 3)
        objs.append(obj)
    return objs


def _fcl(monkeypatch, fn):
    with monkeypatch.context() as m:
        m.setattr(forgeo, "LIB", None)
        return fn()


def test_forgeo_matches_fcl_on_random_poses(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    colliders = _scatter(rng, 8, extent=2.5)
    col = ccol.collision_set(colliders)
    col_fcl = _fcl(monkeypatch, lambda: ccol.collision_set(colliders))

    hits = 0
    for probe in _scatter(rng, 40, extent=3.5):
        got = ccol.intersection_test(col, probe)
        assert got == _fcl(monkeypatch, lambda: ccol.intersection_test(col_fcl, probe))
        hits += got
    # both outcomes must occur, or agreement proves nothing
    assert 5 < hits < 35

    box_hits = 0
    for _ in range(40):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        transform[:3, 3] = rng.uniform(-3.5, 3.5, 3)
        size = tuple(rng.uniform(0.2, 2.0, 3))
        got = ccol.box_intersection_test(col, transform, size)
        assert got == _fcl(
            monkeypatch, lambda: ccol.box_intersection_test(col_fcl, transform, size)
        )
        box_hits += got
    assert 5 < box_hits < 35

    origins = rng.uniform(-8, 8, (200, 3))
    directions = rng.normal(size=(200, 3))
    loc, ray, _ = ccol.raycast(col, origins, directions)
    loc_fcl, ray_fcl, _ = _fcl(
        monkeypatch, lambda: ccol.raycast(col_fcl, origins, directions)
    )
    order, order_fcl = np.argsort(ray), np.argsort(ray_fcl)
    np.testing.assert_array_equal(ray[order], ray_fcl[order_fcl])
    np.testing.assert_allclose(loc[order], loc_fcl[order_fcl], atol=1e-4)
    assert 10 < len(ray) < 190


def test_forgeo_matches_fcl_on_pairwise_self_collision(monkeypatch) -> None:
    rng = np.random.default_rng(1)
    objs = _scatter(rng, 10, extent=3.0)
    agree = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            pair = [objs[i], objs[j]]
            got = ccol.any_self_collision(ccol.collision_set(pair))
            want = _fcl(
                monkeypatch, lambda: ccol.any_self_collision(ccol.collision_set(pair))
            )
            assert got == want, (pair[0].item().name, pair[1].item().name)
            agree.append(got)
    assert 0 < sum(agree) < len(agree)


def test_forgeo_containment_follows_convexity() -> None:
    # a small cube inside a big one: hidden from the hollow-surface test, found
    # because a cube is convex — fcl.Convex behaviour
    outer = pf.ops.primitives.mesh_cube()
    outer.item().location = (0, 0, 0)
    col = ccol.collision_set([outer])
    inner = pf.ops.primitives.mesh_cube(size=0.2)
    inner.item().location = (0.3, -0.2, 0.1)
    assert ccol.intersection_test(col, inner)
    # inside a torus' hole touches nothing, and a torus is hollow anyway
    torus = pf.ops.primitives.mesh_torus()
    col = ccol.collision_set([torus])
    assert not ccol.intersection_test(col, inner)
