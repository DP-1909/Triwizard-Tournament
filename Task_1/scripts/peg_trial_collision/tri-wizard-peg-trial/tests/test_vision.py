"""Vision on synthetic RGB-D frames: right shape, position within 3 mm."""
import math

import numpy as np
import pytest

from peg_trial.utils.synthetic import make_scene
from peg_trial.vision.scene import fuse, process_frame
from peg_trial.vision.shape_detection import classify_shape
from peg_trial.utils.synthetic import polygon

from synthetic_cfg import CAMERAS, COLORS, task

TASK = task()


@pytest.mark.parametrize("shape", ["triangle", "square", "hexagon", "circle"])
@pytest.mark.parametrize("yaw", [0.0, 0.4, 1.1])
def test_classify_ideal_shapes(shape, yaw):
    pts = (polygon(shape, 100, 100, 60, yaw)).astype(np.float32).reshape(-1, 1, 2)
    assert classify_shape(pts)[0] == shape


@pytest.mark.parametrize("seed", range(5))
def test_two_camera_scene(seed):
    results = []
    for cam, Tm in CAMERAS.items():
        T = np.array(Tm, float)
        bgr, depth, K, truth = make_scene(T, table_z=TASK["table_z"], seed=seed, noise_mm=2.0)
        results.append(process_frame(bgr, depth, K, T, COLORS, TASK, cam))
    scene = fuse(results, TASK["vision"]["fuse_merge_dist"])
    assert len(scene["pegs"]) == 4 and len(scene["sockets"]) == 4
    for t in truth["pegs"]:
        d = next(p for p in scene["pegs"] if p["color"] == t["color"])
        assert d["shape"] == t["shape"]
        assert math.hypot(d["x"] - t["x"], d["y"] - t["y"]) < 0.003
        assert abs(d["z_top"] - t["z_top"]) < 0.005
    for t in truth["sockets"]:
        d = min(scene["sockets"], key=lambda s: math.hypot(s["x"] - t["x"], s["y"] - t["y"]))
        assert d["shape"] == t["shape"]
        assert math.hypot(d["x"] - t["x"], d["y"] - t["y"]) < 0.003
