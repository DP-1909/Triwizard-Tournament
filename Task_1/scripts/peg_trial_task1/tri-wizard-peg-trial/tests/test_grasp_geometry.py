"""Random peg rotation: opening width and squeeze-grasp yaw prediction."""
import math

import pytest

from peg_trial.planning.grasp_geometry import (held_width, open_width_needed, predicted_socket_error,
                                               width_along, yaw_after_squeeze)


def test_square_width_depends_on_rotation():
    s = 0.045 * math.sqrt(2)                       # 45 mm square -> corner-to-corner size
    assert width_along("square", s, math.pi / 4, 0.0) == pytest.approx(0.045, abs=1e-6)    # face-on
    assert width_along("square", s, 0.0, 0.0) == pytest.approx(0.0636, abs=1e-3)           # corner-on
    assert held_width("square", s) == pytest.approx(0.045, abs=1e-6)


@pytest.mark.parametrize("yaw", [0.0, 0.3, 0.9, 1.4, 2.5])
def test_open_width_never_smaller_than_held(yaw):
    for shape in ("square", "hexagon", "triangle", "circle"):
        peg = dict(shape=shape, x=0.15, y=0.08, yaw=yaw, size=0.05)
        assert open_width_needed(peg) >= held_width(shape, 0.05) - 1e-9


def test_square_after_squeeze_has_faces_on_fingers():
    peg = dict(shape="square", x=0.15, y=0.0, yaw=0.2, size=0.06)     # heading 0 -> fingers along y
    y = yaw_after_squeeze(peg)
    assert width_along("square", 0.06, y, math.pi / 2) == pytest.approx(held_width("square", 0.06), abs=1e-9)


def test_socket_error_zero_when_socket_turned_with_the_arm():
    peg = dict(shape="hexagon", x=0.15, y=0.05, yaw=0.4, size=0.05)
    hd = math.atan2(-0.08, 0.15 - 0.012)
    sock = dict(x=0.15, y=-0.08, yaw=hd % (math.pi / 3))              # hex corner along its heading
    assert abs(predicted_socket_error(peg, sock)) < 1e-9
