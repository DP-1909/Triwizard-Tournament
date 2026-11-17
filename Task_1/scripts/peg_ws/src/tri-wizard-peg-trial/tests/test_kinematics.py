"""FK and IK must agree: IK(pose) -> joints -> FK(joints) == pose."""
import math
import random

import pytest

from peg_trial.planning.kinematics import IKError, Pose, fk, ik


def test_home_matches_omx_datasheet():
    x, y, z, _ = fk([0, 0, 0, 0])
    assert x == pytest.approx(0.286, abs=1e-3) and y == 0 and z == pytest.approx(0.2045, abs=1e-3)


def test_ik_fk_roundtrip():
    random.seed(1)
    n = 0
    for _ in range(3000):
        p = Pose(random.uniform(0.05, 0.3), random.uniform(-0.25, 0.25), random.uniform(-0.1, 0.25),
                 random.uniform(0, math.pi / 2))
        try:
            q = ik(p)
        except IKError:
            continue
        x, y, z, pitch = fk(q)
        assert (x, y, z, pitch) == pytest.approx((p.x, p.y, p.z, p.pitch), abs=1e-6)
        n += 1
    assert n > 300


def test_unreachable_raises():
    with pytest.raises(IKError):
        ik(Pose(0.6, 0, 0.0))
