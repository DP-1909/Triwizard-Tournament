"""Small helpers to load the YAML config files."""
from __future__ import annotations

import os

import yaml


def config_dir() -> str:
    """config/ folder: env var PEG_TRIAL_CONFIG, else installed share dir, else repo."""
    env = os.environ.get("PEG_TRIAL_CONFIG")
    if env:
        return env
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("peg_trial"), "config")
    except Exception:
        return os.path.join(os.path.dirname(__file__), "..", "..", "config")


def load(name: str, path: str | None = None) -> dict:
    path = path or os.path.join(config_dir(), name)
    with open(path) as f:
        return yaml.safe_load(f)


def load_all(colors=None, calib=None, task=None):
    return load("colors.yaml", colors), load("camera_calib.yaml", calib), load("task.yaml", task)
