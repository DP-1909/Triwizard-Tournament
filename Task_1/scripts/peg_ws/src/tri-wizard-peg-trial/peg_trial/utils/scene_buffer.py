"""Keeps the last N scenes and returns a stable (median) estimate.

Single camera frames jitter by a few mm; taking the median of ~10 frames
before every pick makes the grasp much more reliable.
"""
from __future__ import annotations

import collections
import math
import threading
import time

import numpy as np

SYM = {"square": 4, "triangle": 3, "hexagon": 6, "circle": 0}


def _circ_median(yaws, n):
    if n == 0:
        return 0.0
    s, c = np.sin(np.array(yaws) * n).sum(), np.cos(np.array(yaws) * n).sum()
    return (math.atan2(s, c) / n) % (2 * math.pi / n)


def _stable(items, key_fn):
    groups = {}
    for it in items:
        groups.setdefault(key_fn(it), []).append(it)
    out = []
    for g in groups.values():
        d = dict(g[-1])
        for k in ("x", "y", "z_top", "size"):
            d[k] = float(np.median([i[k] for i in g]))
        d["yaw"] = _circ_median([i["yaw"] for i in g], SYM.get(d["shape"], 0))
        # most frequently seen shape wins (protects against one bad frame)
        d["shape"] = collections.Counter(i["shape"] for i in g).most_common(1)[0][0]
        d["n"] = len(g)
        out.append(d)
    return out


def _socket_key(socks, dist=0.015):
    """Greedy clustering: sockets within `dist` m of a cluster centre = same socket."""
    centres = []
    keys = {}
    for s in socks:
        for i, (cx, cy) in enumerate(centres):
            if math.hypot(s["x"] - cx, s["y"] - cy) < dist:
                keys[id(s)] = i
                break
        else:
            centres.append((s["x"], s["y"]))
            keys[id(s)] = len(centres) - 1
    return lambda s: keys[id(s)]


class SceneBuffer:
    def __init__(self, size=15):
        self.buf = collections.deque(maxlen=size)
        self.lock = threading.Lock()

    def add(self, scene: dict):
        with self.lock:
            self.buf.append((time.time(), scene))

    def clear(self):
        with self.lock:
            self.buf.clear()

    def snapshot(self, min_frames=5, timeout=5.0, max_age=2.0, expected_sockets=4):
        """Wait until `min_frames` fresh scenes exist, return a median scene (or None)."""
        t0 = time.time()
        last_scene = None
        while time.time() - t0 < timeout:
            with self.lock:
                fresh = [s for t, s in self.buf if time.time() - t < max_age]
            if len(fresh) >= min_frames:
                pegs = [p for s in fresh for p in s["pegs"]]
                socks = [s_ for s in fresh for s_ in s["sockets"]]
                stable_pegs = [p for p in _stable(pegs, lambda p: p["color"]) if p["n"] >= min_frames // 2]
                stable_socks = [s for s in _stable(socks, _socket_key(socks)) if s["n"] >= min_frames // 2]
                last_scene = {"pegs": stable_pegs, "sockets": stable_socks}
                if len(stable_socks) >= expected_sockets:
                    return last_scene
            time.sleep(0.05)
        return last_scene
