"""Fixed configs for the synthetic tests, so they do not depend on your tuned lab configs
(your real colours.yaml / camera_calib.yaml / table_z can change freely)."""
import copy

from peg_trial.utils.config import load

COLORS = {
    "morphology": {"open": 5, "close": 7},
    "pegs": {
        "red": {"hsv": [[[0, 120, 70], [8, 255, 255]], [[170, 120, 70], [179, 255, 255]]], "shape": ""},
        "green": {"hsv": [[[40, 80, 50], [85, 255, 255]]], "shape": ""},
        "blue": {"hsv": [[[95, 120, 50], [130, 255, 255]]], "shape": ""},
        "yellow": {"hsv": [[[20, 120, 100], [35, 255, 255]]], "shape": ""},
    },
    "board": {"hsv": [[[0, 0, 160], [179, 40, 255]]]},
}
CAMERAS = {
    "cam1": [[0.0, -1.0, 0.0, 0.20], [-1.0, 0.0, 0.0, 0.10], [0.0, 0.0, -1.0, 0.55], [0, 0, 0, 1]],
    "cam2": [[0.0, -1.0, 0.0, 0.20], [-1.0, 0.0, 0.0, -0.10], [0.0, 0.0, -1.0, 0.55], [0, 0, 0, 1]],
}


def task(table_z=-0.05, peg_height=0.10):
    t = copy.deepcopy(load("task.yaml"))
    t["table_z"] = table_z
    t["peg_height"] = peg_height
    t["vision"]["min_socket_area_px"] = 500
    t.get("place", {})["circle_y_offset"] = 0.0
    t.get("place", {})["hexagon_y_offset"] = 0.0
    t.get("place", {})["square_y_offset"] = 0.0
    t.get("place", {})["square_x_offset"] = 0.0
    t.get("grasp", {})["hexagon_grasp_backward_offset"] = 0.0
    t.get("grasp", {})["square_grasp_backward_offset"] = 0.0
    t.get("grasp", {})["grasp_depth"] = 0.004
    t.get("grasp", {})["carry_z"] = 0.113
    return t
