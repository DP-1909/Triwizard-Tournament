"""Collision-free paths for the gripper at a constant height (x-y plane).

Idea (beginner version):
  1. Lay a grid (8 mm cells) over the robot's reachable ring.
  2. A cell is FREE if (a) IK can reach it with the gripper pointing down at
     the travel height and (b) the gripper (+ carried peg) placed there does
     not touch anything (collision.check_state).
  3. A* search finds the shortest chain of free cells from start to goal.
  4. "Shortcutting" removes zig-zags: if two far-apart points can be joined by a
     straight free line, the points in between are dropped.
The result is a few straight segments; the planner turns each into small
Cartesian IK steps, so the real gripper follows exactly the checked path.
"""
from __future__ import annotations

import heapq
import math

from .collision import check_state
from .kinematics import IKError, J1_X, Pose, ik

DOWN = math.pi / 2


def segment_free(a, b, z, gap, world, model, holding, step=0.005, pitch=DOWN):
    """True if every point on the straight line a->b (x, y) at height z is reachable and free."""
    n = max(1, int(math.ceil(math.dist(a, b) / step)))
    for i in range(n + 1):
        x = a[0] + (b[0] - a[0]) * i / n
        y = a[1] + (b[1] - a[1]) * i / n
        try:
            ik(Pose(x, y, z, pitch))
        except IKError:
            return False
        if check_state(x, y, z, gap, world, model, holding):
            return False
    return True


def astar_xy(start, goal, z, gap, world, model, holding=None, res=0.008,
             r_min=0.09, r_max=0.27, max_nodes=6000):
    """Grid A* in the x-y plane at height z. Returns [start, ..., goal] or None."""
    def key(p):
        return (round(p[0] / res), round(p[1] / res))

    def to_xy(k):
        return (k[0] * res, k[1] * res)

    free_cache = {}

    def free(k):
        if k not in free_cache:
            x, y = to_xy(k)
            r = math.hypot(x - J1_X, y)
            ok = r_min <= r <= r_max
            if ok:
                try:
                    ik(Pose(x, y, z, DOWN))
                except IKError:
                    ok = False
            if ok:
                ok = check_state(x, y, z, gap, world, model, holding) is None
            free_cache[k] = ok
        return free_cache[k]

    s, g = key(start), key(goal)
    if not free(s) or not free(g):
        return None
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    openq = [(0.0, s)]
    cost = {s: 0.0}
    parent = {s: None}
    while openq and len(cost) < max_nodes:
        _, cur = heapq.heappop(openq)
        if cur == g:
            break
        for dx, dy in nbrs:
            nx = (cur[0] + dx, cur[1] + dy)
            if not free(nx):
                continue
            c = cost[cur] + math.hypot(dx, dy)
            if c < cost.get(nx, 1e18):
                cost[nx] = c
                parent[nx] = cur
                heapq.heappush(openq, (c + math.hypot(g[0] - nx[0], g[1] - nx[1]), nx))
    if g not in parent:
        return None
    path, k = [], g
    while k is not None:
        path.append(to_xy(k))
        k = parent[k]
    path.reverse()
    path[0], path[-1] = tuple(start), tuple(goal)
    path = shortcut(path, z, gap, world, model, holding)
    # final safety net: every segment must be free at the fine step the robot will follow
    if not all(segment_free(a, b, z, gap, world, model, holding, step=0.004) for a, b in zip(path[:-1], path[1:])):
        return None
    return path


def shortcut(path, z, gap, world, model, holding):
    """Greedy smoothing: from each point jump to the farthest point reachable in a straight free line."""
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not segment_free(path[i], path[j], z, gap, world, model, holding):
            j -= 1
        out.append(path[j])
        i = j
    return out


def plan_xy(start, goal, z, gap, world, model, holding=None):
    """Straight line if it is free, otherwise A* around the obstacles."""
    if segment_free(start, goal, z, gap, world, model, holding):
        return [tuple(start), tuple(goal)]
    return astar_xy(start, goal, z, gap, world, model, holding)
