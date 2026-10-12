# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import time

import bpy
import numpy as np

from infinigen2.objects import vase


def test_cup_subdivision_is_fast_with_many_ridges() -> None:
    # seed 15 draws the maximum 80 ridges, so the floor is a single 320-gon
    obj = vase.cup_rand(np.random.default_rng(15)).mesh.item()
    assert max(p.loop_total for p in obj.data.polygons) == 320

    obj.modifiers["SUBSURF"].show_viewport = True
    start = time.perf_counter()
    bpy.context.view_layer.update()
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
