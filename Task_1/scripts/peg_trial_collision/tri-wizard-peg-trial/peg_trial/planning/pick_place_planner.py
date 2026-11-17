"""Pick-and-place planner: one peg + its socket -> list of robot steps.

Sequence (matches the task sheet's "Control and planning" rules), now COLLISION-AWARE:

  home
   -> TRAVEL     : to above the peg at a safe height; around other pegs if needed (A*)
   -> PRE-GRASP  : `pregrasp_offset` above the peg top, gripper partly open
   -> APPROACH   : straight down along the board normal to the grasp height
   -> SETTLE     : wait `settle_delay`, then CLOSE, wait `close_delay`
   -> LIFT       : straight up to the carry height
   -> TRANSIT    : over the other pegs if the arm can reach that high, else AROUND them (A*)
   -> HOVER      : `hover_offset` above the socket
   -> INSERT     : lower SLOWLY in small steps to the insert depth
   -> RELEASE    : open a little, wait, straight up, go home

Every pose of every move is checked against the other pegs, the pegs already
placed, the board and the table (see collision.py). If no safe plan exists a
PlanError says exactly what would be hit, and the autonomy node tries another peg first.

Each step is a dict:
   {"type": "move", "name": ..., "points": [(q[4], t_from_start), ...]}
   {"type": "gripper", "name": ..., "cmd": "open" | "close"}
   {"type": "wait", "name": ..., "sec": float}
"""
from __future__ import annotations

import math

from .collision import GripperModel, build_world, check_state
from .grasp_geometry import held_width, open_width_needed, predicted_socket_error, with_known_size
from .kinematics import IKError, Pose, fk, ik, ik_best, interpolate_line
from .path_planner import plan_xy

DOWN = math.pi / 2
SHAPE_SYM = {"square": 4, "triangle": 3, "hexagon": 6, "circle": 0}


class PlanError(Exception):
    pass


def match_pegs_to_sockets(pegs, sockets):
    """Same shape = match. If two sockets have the same shape, take the closest."""
    pairs, used = [], set()
    for p in pegs:
        cands = [(i, s) for i, s in enumerate(sockets) if s["shape"] == p["shape"] and i not in used]
        if not cands:
            continue
        i, s = min(cands, key=lambda c: math.hypot(c[1]["x"] - p["x"], c[1]["y"] - p["y"]))
        used.add(i)
        pairs.append((p, s))
    return pairs


def yaw_error(peg, sock, sym):
    """Peg yaw after transport minus socket yaw, IGNORING the squeeze (kept for reference;
    the planner uses grasp_geometry.predicted_socket_error).

    The OMX has no wrist roll: the peg turns WITH joint1. So if joint1 turns
    by d = atan2(socket) - atan2(peg), the peg's yaw also changes by d.
    """
    if sym == 0:
        return 0.0
    d = math.atan2(sock["y"], sock["x"] - 0.012) - math.atan2(peg["y"], peg["x"] - 0.012)
    period = 2 * math.pi / sym
    e = (peg["yaw"] + d - sock["yaw"]) % period
    return e - period if e > period / 2 else e


def _joint_time(q0, q1, max_vel=0.8, min_t=0.6):
    return max(min_t, max(abs(a - b) for a, b in zip(q0, q1)) / max_vel)


def _line(p0, p1, step, speed, min_t, t0, strict=True, pitch_tol=0.0):
    """IK every few mm along a straight line -> joint points with timestamps."""
    pts, t = [], t0
    poses = interpolate_line(p0, p1, step)
    seg_t = max(min_t, math.dist((p0.x, p0.y, p0.z), (p1.x, p1.y, p1.z)) / speed) / len(poses)
    for p in poses:
        q = ik(p) if strict else ik_best(p, pitch_tol)
        t += seg_t
        pts.append((q, t))
    return pts


# --------------------------------------------------------------------------- helpers
def _reachable(x, y, z):
    try:
        ik(Pose(x, y, z, DOWN))
        return True
    except IKError:
        return False


def max_reachable_z(x, y, target_z=0.125, min_z=0.020, step=0.001):
    """Find the highest reachable Z at (x, y) with top-down gripper (DOWN), up to target_z."""
    if _reachable(x, y, target_z):
        return target_z
    z = target_z
    while z >= min_z:
        if _reachable(x, y, z):
            return round(z, 4)
        z -= step
    return None


def _checked_line(p0, p1, step, speed, min_t, gap, world, model, holding, ignore_board=False, what=""):
    """Straight Cartesian line, gripper pointing down; every IK point collision-checked."""
    poses = interpolate_line(p0, p1, step)
    seg_t = max(min_t, math.dist((p0.x, p0.y, p0.z), (p1.x, p1.y, p1.z)) / speed) / len(poses)
    pts, t = [], 0.0
    for p in poses:
        try:
            q = ik(p)
        except IKError as e:
            raise PlanError(f"{what}: ({p.x:.3f},{p.y:.3f},{p.z:.3f}) unreachable: {e}") from e
        hit = check_state(p.x, p.y, p.z, gap, world, model, holding, ignore_board)
        if hit:
            raise PlanError(f"{what}: {hit} at ({p.x:.3f},{p.y:.3f},{p.z:.3f})")
        t += seg_t
        pts.append((q, t))
    return pts


def _xy_path_moves(name, start, goal, z, gap, world, model, holding, step, speed, min_t):
    """Collision-free horizontal path at height z -> list of move steps (one per straight segment)."""
    path = plan_xy((start.x, start.y), (goal.x, goal.y), z, gap, world, model, holding)
    if path is None:
        raise PlanError(f"{name}: no collision-free path at z={z:.3f} "
                        f"(pegs too close together, or raise the robot — see task.yaml)")
    moves = []
    for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
        moves.append({"type": "move", "name": f"{name}{'' if len(path) == 2 else f'_{i + 1}'}",
                      "points": _checked_line(Pose(a[0], a[1], z), Pose(b[0], b[1], z), min(step, 0.004), speed,
                                              min_t, gap, world, model, holding, what=name)})
    return moves


def _vertical(p, z, step, speed, min_t, gap, world, model, holding, name, ignore_board=False):
    """Straight up/down to height z (no step if already there)."""
    if abs(p.z - z) < 1e-4:
        return []
    return [{"type": "move", "name": name,
             "points": _checked_line(p, Pose(p.x, p.y, z), step, speed / 2, min_t, gap, world, model, holding,
                                     ignore_board, what=name)}]


def _ready_pose(cfg, world, model, gap):
    """Top-down pose near the base, as high as the arm reaches and clear of every peg.
    Every pick starts and ends here, so the joint-space jump from/to HOME happens in free space."""
    rx, ry = cfg["motion"].get("ready_xy", [0.125, 0.0])
    r0, a0 = math.hypot(rx - 0.012, ry), math.atan2(ry, rx - 0.012)
    for da in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
        for r in (r0, r0 - 0.015, r0 + 0.015):
            a = a0 + math.radians(da)
            x, y = 0.012 + r * math.cos(a), r * math.sin(a)
            z = next((z / 1000 for z in range(140, 40, -2) if _reachable(x, y, z / 1000)), None)
            if z is not None and check_state(x, y, z, gap, world, model) is None:
                return Pose(x, y, z)
    raise PlanError("no free READY pose near the base — clear the area around the robot")


def _checked_joint_move(q0, q1, world, model, gap, holding=None, what="joint move", samples=20):
    """Joint-space move; sample the motion and check the gripper wherever it points roughly down."""
    for i in range(1, samples + 1):
        q = [a + (b - a) * i / samples for a, b in zip(q0, q1)]
        x, y, z, pitch = fk(q)
        if abs(pitch - DOWN) < 0.5:                      # model is valid for a near-vertical gripper
            hit = check_state(x, y, z, gap, world, model, holding)
            if hit:
                raise PlanError(f"{what}: {hit}")
        elif z < world.table_z + 0.03:
            raise PlanError(f"{what}: gripper too close to the table")
    return [(q1, _joint_time(q0, q1))]



def _chain_points_3d(q):
    """3-D points of the arm links for joint vector q: [link3 start, wrist, TCP, fingertip]."""
    from .kinematics import A_X, A_Z, J1_X, J2_Z, LB, LC
    q1, q2, q3, q4 = q
    f = q2 + q3 + q4
    r2, h2 = A_X * math.cos(q2) + A_Z * math.sin(q2), J2_Z + (-A_X * math.sin(q2) + A_Z * math.cos(q2))
    rw, hw = r2 + LB * math.cos(q2 + q3), h2 - LB * math.sin(q2 + q3)
    rt, ht = rw + LC * math.cos(f), hw - LC * math.sin(f)
    rf, hf = rt + 0.021 * math.cos(f), ht - 0.021 * math.sin(f)
    return [(J1_X + r * math.cos(q1), r * math.sin(q1), h) for r, h in ((r2, h2), (rw, hw), (rt, ht), (rf, hf))]


def _swing_clear(q0, q1, pegs, top_z, margin=0.025, samples=60, what="joint swing"):
    """Joint-space move: make sure no arm link / fingertip passes through the volume around ANY peg
    (cylinder radius + margin, from the table up to the peg top). Works for any gripper pitch."""
    cyl = [(p["x"], p["y"], max(0.015, p.get("size", 0.05) / 2) + margin) for p in pegs]
    for i in range(1, samples + 1):
        q = [a + (b - a) * i / samples for a, b in zip(q0, q1)]
        pts = _chain_points_3d(q)
        for (ax, ay, az), (bx, by, bz) in zip(pts[:-1], pts[1:]):
            for j in range(13):
                u = j / 12
                px, py, pz = ax + (bx - ax) * u, ay + (by - ay) * u, az + (bz - az) * u
                for cx, cy, cr in cyl:
                    if math.hypot(px - cx, py - cy) < cr and pz < top_z + 0.004:
                        raise PlanError(f"{what}: arm would sweep through a peg at ({cx:.3f},{cy:.3f}) "
                                        f"(point z={pz:.3f} < peg top {top_z:.3f})")


def _span(z0, z1, step=0.004):
    lo, hi = min(z0, z1), max(z0, z1)
    n = max(1, int((hi - lo) / step))
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


def choose_grasp(peg, sock, cfg, world, model, peg_h, gap_open=0.056, gap_hold=0.045, radius=0.0225):
    """Pick the grasp depth and the heights.

    The OMX can only reach LOW at the far sockets with a straight-down gripper
    (r=0.20 m -> TCP z <= 0.098 m, r=0.23 m -> z <= 0.058 m). A 100 mm peg held only 18 mm below its top
    hangs 82 mm under the TCP, so its bottom cannot get down to the socket. The grasp therefore has to get
    DEEPER the further the socket is. This function tries depths from grasp_depth up to max_grasp_depth
    in 2 mm steps and returns the SHALLOWEST depth for which every pose is reachable and collision-free.

    Heights (TCP z, base frame):
      z_final = socket top - insert_depth + (peg_h - d)    exact insertion, peg bottom insert_depth inside the socket
      z_hover = socket top + hover_offset + (peg_h - d)    capped by reach, but never lower than +4 mm clearance
      z_carry = task carry_z, capped by the reach at BOTH peg and socket
    """
    g, pl = cfg["grasp"], cfg["place"]
    d_pref = g["grasp_depth"]
    d_board = peg_h - pl["insert_depth"] - model.tip_below - 0.004   # fingertips stay above the board at insertion
    d_max = max(d_pref, min(g.get("max_grasp_depth", 0.050), d_board))
    clear = g["lift_clearance"]
    min_clear = pl.get("min_hover_clearance", 0.004)                 # peg bottom stays >= this above the socket top
    board_top = max(sock.get("z_top", 0.0), cfg["table_z"] + pl.get("board_height", 0.008))
    tops = [c.z1 for c in world.cylinders]
    top_all = max(tops) if tops else cfg["table_z"]
    n_cand = int(round((d_max - d_pref) / 0.002)) + 1
    candidates = [d_pref + 0.002 * i for i in range(n_cand)]
    offsets = [g["pregrasp_offset"]] + [o for o in (0.015, 0.010, 0.006) if o < g["pregrasp_offset"]]
    reach_peg = max_reachable_z(peg["x"], peg["y"], target_z=0.30)
    reach_sock = max_reachable_z(sock["x"], sock["y"], target_z=0.30)
    if reach_peg is None or reach_sock is None:
        raise PlanError("peg or socket is outside the robot's top-down workspace")

    def heights(d, over):
        """-> (dict, None) or (None, reason)"""
        below = peg_h - d                                          # TCP -> peg bottom
        z_grasp = peg["z_top"] - d
        z_final = sock["z_top"] - pl["insert_depth"] + below
        z_hover_min = sock["z_top"] + min_clear + below
        if over:
            z_carry = max(top_all + clear + below, z_grasp + 0.01)
        else:
            cand = cfg.get("table_z", 0.0) + g["carry_z"] if "carry_z" in g else board_top + clear + below
            z_carry = min(cand, reach_peg, reach_sock)
            if z_carry < max(z_hover_min, z_grasp + 0.01):
                return None, (f"carry z={z_carry:.3f} too low: socket reach is {reach_sock:.3f}, "
                              f"hover needs {z_hover_min:.3f}")
        z_hover = min(sock["z_top"] + pl["hover_offset"] + below, z_carry, reach_sock)
        if z_hover < z_hover_min - 1e-6:
            return None, f"socket too far: hover needs z={z_hover_min:.3f}, reach is {reach_sock:.3f}"
        need = [(peg["x"], peg["y"], z_grasp), (peg["x"], peg["y"], z_carry),
                (sock["x"], sock["y"], z_carry), (sock["x"], sock["y"], z_hover), (sock["x"], sock["y"], z_final)]
        names = ["z_grasp", "z_carry_peg", "z_carry_sock", "z_hover", "z_final"]
        for nm, n in zip(names, need):
            if not _reachable(*n):
                return None, f"unreachable @ {nm} ({n[2]:.3f})"
        z_pre = next((peg["z_top"] + o for o in offsets if _reachable(peg["x"], peg["y"], peg["z_top"] + o)), None)
        if z_pre is None:
            return None, "pre-grasp unreachable"
        hold = (peg_h, d, radius)
        checks = [(peg["x"], peg["y"], z, gap_open, None, False) for z in _span(z_grasp, z_pre)]
        checks += [(peg["x"], peg["y"], z, gap_hold, hold, False) for z in _span(z_grasp + 0.004, z_carry)]
        checks += [(sock["x"], sock["y"], z, gap_hold, hold, True) for z in _span(z_final, z_hover)]
        for x, y, z, gp, hd, ib in checks:
            hit = check_state(x, y, z, gp, world, model, hd, ib)
            if hit:
                return None, f"collision @ ({x:.3f},{y:.3f},z={z:.3f}): {hit}"
        return dict(d=d, z_grasp=z_grasp, z_pre=z_pre, z_carry=z_carry, z_hover=z_hover,
                    z_final=z_final, over=over), None

    # preference: 1) over the pegs while still gripping high (<= preferred + 1 cm)
    #             2) around the pegs at the shallowest feasible grip   3) over, gripping lower
    near = d_pref + g.get("over_extra_depth", 0.010)
    diag = []
    for over, lo, hi in ((True, 0, near), (False, 0, d_max), (True, near, d_max)):
        if over and not tops:
            continue
        for d in candidates:
            if d > min(hi, d_max) + 1e-9 or (lo > 0 and d <= lo + 1e-9):
                continue
            h, why = heights(d, over)
            if h:
                return h
            if len(diag) < 40:
                diag.append(f"d={d * 1000:.0f}mm over={over}: {why}")
    raise PlanError(f"no grasp depth between {d_pref * 1000:.0f} and {d_max * 1000:.0f} mm is both reachable and "
                    f"collision-free for peg r={math.hypot(peg['x'] - 0.012, peg['y']):.3f} / socket "
                    f"r={math.hypot(sock['x'] - 0.012, sock['y']):.3f} (top-down reach at socket: "
                    f"z<={reach_sock:.3f}) | " + " | ".join(diag[:2] + diag[-2:]))


# --------------------------------------------------------------------------- main planner
def plan_pick_place(peg, sock, cfg, start_q=None, obstacles=None, placed=None, sockets=None, return_home=True):
    """Collision-aware plan for one peg.

    obstacles : other pegs still on the table (vision dicts)
    placed    : [(peg, socket), ...] already inserted pegs
    sockets   : all sockets (to model the board)
    """
    g, pl, mo = cfg["grasp"], cfg["place"], cfg["motion"]
    ccfg = cfg.get("collision", {})
    step, speed, min_t = mo["cartesian_step"], mo["move_speed"], mo["min_segment_time"]
    home = mo["home_joints"]
    start_q = list(start_q or home)

    peg_h = peg["z_top"] - cfg["table_z"]
    if abs(peg_h - cfg["peg_height"]) > 0.015:          # bad depth reading -> trust the known height
        peg_h = cfg["peg_height"]
    peg = with_known_size(dict(peg, z_top=cfg["table_z"] + peg_h), cfg)     # exact measured peg size
    sh_p = (peg.get("shape", "")).lower().strip()
    if sh_p == "circle":
        peg["x"] += g.get("circle_grasp_x_offset", 0.0)
        peg["y"] += g.get("circle_grasp_y_offset", 0.0)
    obstacles = [with_known_size(o, cfg) for o in (obstacles or [])]
    placed = [(with_known_size(p, cfg), s) for p, s in (placed or [])]

    model = GripperModel(ccfg)
    world = build_world(cfg, obstacles or [], placed or [], sockets or [], peg_h)
    # ---- finger openings, adapted to this peg's RANDOM yaw (see grasp_geometry.py)
    max_gap = cfg["robot"].get("gripper_gap_open", ccfg.get("pregrasp_gap", 0.080))   # widest possible gap
    need = open_width_needed(peg)                          # peg width along the finger axis right now
    clear = ccfg.get("open_clearance", 0.006)              # per side
    if need + 2 * 0.002 > max_gap:
        raise PlanError(f"peg is {need * 1000:.0f} mm wide across the fingers at this rotation; "
                        f"gripper opens only {max_gap * 1000:.0f} mm")
    gap_open = min(max_gap, need + 2 * clear)
    radius = max(0.015, peg.get("size", 0.05) / 2)
    gap_hold = min(gap_open, held_width(peg["shape"], peg.get("size", 0.05)))   # after the squeeze
    gap_release = min(gap_open, gap_hold + ccfg.get("release_extra_gap", 0.008))   # open just a little to let go
    holding = (peg_h, None, radius)                        # grasp depth filled in below

    h = choose_grasp(peg, sock, cfg, world, model, peg_h, gap_open, gap_hold, radius)
    d = h["d"]
    holding = (peg_h, d, radius)
    warnings = []
    if gap_open < need + 2 * clear - 1e-6:
        warnings.append(f"only {(gap_open - need) * 500:.0f} mm finger clearance per side for this peg rotation")

    pre = Pose(peg["x"], peg["y"], h["z_pre"])
    grasp = Pose(peg["x"], peg["y"], h["z_grasp"])
    carry_peg = Pose(peg["x"], peg["y"], h["z_carry"])
    carry_sock = Pose(sock["x"], sock["y"], h["z_carry"])
    hover = Pose(sock["x"], sock["y"], h["z_hover"])
    final = Pose(sock["x"], sock["y"], h["z_final"])
    placed_top = sock["z_top"] - pl["insert_depth"] + peg_h
    world_after = build_world(cfg, obstacles or [], list(placed or []) + [(peg, sock)], sockets or [], peg_h)
    world_wo_peg = build_world(cfg, obstacles or [], placed or [], sockets or [], peg_h)   # without the new one
    world_travel = build_world(cfg, list(obstacles or []) + [peg], placed or [], sockets or [], peg_h)
    ready = _ready_pose(cfg, world_after, model, gap_open)
    clear_tip = model.tip_below + model.margin + 0.002       # fingertips this far above a peg top

    def travel_z(wld, pts, target_top):
        """Empty-gripper flying height: over EVERY peg if reachable (straight line), else just over the
        target/placed peg and steer AROUND the others."""
        tops = [c.z1 for c in wld.cylinders]
        for z in ((max(tops) if tops else cfg["table_z"]) + clear_tip, target_top + clear_tip):
            if all(_reachable(x, y, z) for x, y in pts):
                return z
        raise PlanError(f"gripper cannot get its fingertips above the peg top here (needs z={target_top + clear_tip:.3f})"
                        f" — peg/socket too far from the robot, or raise the robot (riser, table_z)")

    steps = [{"type": "gripper", "name": "open", "cmd": "open", "gap": gap_open}]

    # ---- 1. travel (empty gripper) to above the peg
    x0, y0, z0, p0 = fk(start_q)
    is_at_home = (abs(p0) < 0.25 and z0 > 0.18)
    if is_at_home:
        # Rotate Joint 1 in the air at the high folded home pose (z ~ 0.24m, > 14cm above any peg)
        theta_peg = math.atan2(pre.y, pre.x - 0.012)
        q_align = [theta_peg, home[1], home[2], home[3]]
        t_rot = max(0.5, abs(theta_peg - start_q[0]) / 0.8)
        steps.append({"type": "move", "name": "align_high", "points": [(q_align, t_rot)]})
        # Unfold directly along the radial azimuth to pre-grasp hovering 20 mm above peg
        q_pre = ik(pre)
        # The unfold from the folded home pose is a joint-space swing (pitch goes from ~-0.2 rad to 90 deg):
        # check every link and the fingertips against EVERY peg, not only where the gripper is vertical.
        _swing_clear(q_align, q_pre, [peg] + list(obstacles or []) + [p for p, _ in (placed or [])],
                     peg["z_top"], what="home -> pregrasp swing")
        steps.append({"type": "move", "name": "to_pregrasp",
                      "points": _checked_joint_move(q_align, q_pre, world, model, gap_open, what="unfold to pregrasp")})
        # fully open (-0.010) while hovering high above the peg, BEFORE the vertical descent
        steps.append({"type": "gripper", "name": "open_pregrasp", "cmd": "open", "gap": gap_open})
        steps.append({"type": "wait", "name": "open_settle", "sec": g.get("open_settle", 0.6)})
    else:
        q_ready = ik(ready)
        steps.append({"type": "move", "name": "to_ready",
                      "points": _checked_joint_move(start_q, q_ready, world_travel, model, gap_open,
                                                    what="start -> ready")})
        x0, y0, z0 = ready.x, ready.y, ready.z
        tz = travel_z(world_travel, [(x0, y0), (pre.x, pre.y)], peg["z_top"])
        steps += _vertical(Pose(x0, y0, z0), tz, step, speed, min_t, gap_open, world_travel, model, None, "rise")
        steps += _xy_path_moves("travel", Pose(x0, y0, tz), Pose(pre.x, pre.y, tz), tz, gap_open, world_travel, model,
                                None, step, speed, min_t)
        steps += _vertical(Pose(pre.x, pre.y, tz), pre.z, step, speed, min_t, gap_open, world, model, None,
                           "down to pre-grasp")

    # ---- 2. approach, grasp, lift
    # Shape-specific pregrasp backward adjustments towards origin:
    sh_peg = (peg.get("shape", "")).lower().strip()
    hex_backward = g.get("hexagon_grasp_backward_offset", 0.0)
    sq_backward = g.get("square_grasp_backward_offset", 0.0)
    tri_backward = g.get("triangle_grasp_backward_offset", 0.0)
    backward_dist = 0.0
    if sh_peg in ("hexagon", "hex"):
        backward_dist = hex_backward
    elif sh_peg in ("square", "sq"):
        backward_dist = sq_backward
    elif sh_peg in ("triangle", "tri"):
        backward_dist = tri_backward

    if backward_dist > 0:
        r_orig = math.hypot(pre.x, pre.y)
        if r_orig > 1e-4:
            dx_back = -backward_dist * (pre.x / r_orig)
            dy_back = -backward_dist * (pre.y / r_orig)
            pre_shifted = Pose(pre.x + dx_back, pre.y + dy_back, pre.z)
            grasp_shifted = Pose(grasp.x + dx_back, grasp.y + dy_back, grasp.z)
            carry_peg_shifted = Pose(carry_peg.x + dx_back, carry_peg.y + dy_back, carry_peg.z)
            steps.append({"type": "move", "name": f"{sh_peg}_pregrasp_adjust",
                          "points": _checked_line(pre, pre_shifted, step, speed / 2, min_t, gap_open, world, model,
                                                  None, ignore_board=True, what=f"{sh_peg} pregrasp adjust")})
            pre = pre_shifted
            grasp = grasp_shifted
            carry_peg = carry_peg_shifted

    steps.append({"type": "move", "name": "approach",
                  "points": _checked_line(pre, grasp, step, speed / 2, min_t, gap_open, world, model, None,
                                          what="approach")})
    steps.append({"type": "wait", "name": "settle", "sec": g["settle_delay"]})
    steps.append({"type": "gripper", "name": "close", "cmd": "close"})
    steps.append({"type": "wait", "name": "grip", "sec": g["close_delay"]})
    lift_target = Pose(grasp.x, grasp.y, max(carry_peg.z, pre.z))   # pure vertical lift to the high hover first
    steps.append({"type": "move", "name": "lift",
                  "points": _checked_line(grasp, lift_target, step, speed / 2, min_t, gap_hold, world, model,
                                          holding, what="lift")})
    if lift_target.z > carry_peg.z + 1e-4:
        steps.append({"type": "move", "name": "to_transit_height",
                      "points": _checked_line(lift_target, carry_peg, step, speed, min_t, gap_hold, world, model,
                                              holding, what="to transit height")})

    # ---- 3. transit (over or around) and insert
    steps += _xy_path_moves("transit", carry_peg, carry_sock, h["z_carry"], gap_hold, world, model, holding,
                            step, speed, min_t)
    if carry_sock.z > hover.z + 1e-4:
        steps.append({"type": "move", "name": "to_hover",
                      "points": _checked_line(carry_sock, hover, step, speed / 2, min_t, gap_hold, world, model,
                                              holding, ignore_board=True, what="to hover")})

    # Shape-specific X and Y offsets: after hovering and before placing, move end effector in XY
    circle_y_off = pl.get("circle_y_offset", 0.0)
    hexagon_y_off = pl.get("hexagon_y_offset", 0.0)
    square_y_off = pl.get("square_y_offset", 0.0)
    square_x_off = pl.get("square_x_offset", 0.0)
    tri_place_back = pl.get("triangle_place_backward_offset", 0.0)
    tri_x_off = pl.get("triangle_x_offset", 0.0)
    tri_y_off = pl.get("triangle_y_offset", 0.0)
    sh_check = (sock.get("shape", "") or peg.get("shape", "")).lower().strip()

    x_shift, y_shift = 0.0, 0.0
    if sh_check == "circle":
        y_shift = circle_y_off
    elif sh_check in ("hexagon", "hex"):
        y_shift = hexagon_y_off
    elif sh_check in ("square", "sq"):
        x_shift = square_x_off
        y_shift = square_y_off
    elif sh_check in ("triangle", "tri"):
        r_sock = math.hypot(hover.x, hover.y)
        if r_sock > 1e-4 and tri_place_back > 0:
            x_shift = -tri_place_back * (hover.x / r_sock)
            y_shift = -tri_place_back * (hover.y / r_sock)
        x_shift += tri_x_off
        y_shift += tri_y_off

    if abs(x_shift) > 1e-6 or abs(y_shift) > 1e-6:
        hover_shifted = Pose(hover.x + x_shift, hover.y + y_shift, hover.z)
        final = Pose(final.x + x_shift, final.y + y_shift, final.z)
        steps.append({"type": "move", "name": f"{sh_check}_adjust",
                      "points": _checked_line(hover, hover_shifted, step, speed / 2, min_t, gap_hold, world, model,
                                              holding, ignore_board=True, what=f"{sh_check} xy-adjust")})
        hover = hover_shifted

    ins, t = [], 0.0
    if hover.z > final.z + 1e-4:
        for p in interpolate_line(hover, final, pl["descend_step"]):
            hit = check_state(p.x, p.y, p.z, gap_hold, world, model, holding, ignore_board=True)
            if hit:
                raise PlanError(f"insert: {hit}")
            t += pl["descend_time_per_step"]
            ins.append((ik(p), t))
    else:
        ins.append((ik(final), pl.get("descend_time_per_step", 0.15)))
    steps.append({"type": "move", "name": "insert", "points": ins})

    # ---- 4. release and leave the placed peg
    steps.append({"type": "gripper", "name": "release", "cmd": "open", "gap": gap_release})
    steps.append({"type": "wait", "name": "release_wait", "sec": pl["release_delay"]})
    clear_z = max(placed_top + model.tip_below + model.margin, final.z + 0.01)
    if _reachable(final.x, final.y, clear_z):
        # straight up until the fingertips are above the placed peg
        steps.append({"type": "move", "name": "retreat",
                      "points": _checked_line(final, Pose(final.x, final.y, clear_z), step, speed / 2, min_t,
                                              gap_release, world_wo_peg, model, None, ignore_board=True,
                                              what="retreat")})
        cur = Pose(final.x, final.y, clear_z)
    else:
        # too far out to lift that high: slide the open fingers back towards the robot along the peg
        r = math.hypot(final.x - 0.012, final.y)
        ux, uy = (final.x - 0.012) / r, final.y / r
        back = max(0.070, model.finger_hx + radius + model.margin + 0.035)
        cur = Pose(final.x - ux * back, final.y - uy * back, final.z)
        steps.append({"type": "move", "name": "retreat_slide",
                      "points": _checked_line(final, cur, step, speed / 3, min_t, gap_release, world_wo_peg, model,
                                              None, ignore_board=True, what="retreat (slide back)")})
    if return_home:
        # If cur is low (e.g. from retreat_slide), first rise vertically clear of the board and peg
        target_rise = min(0.125, max(cur.z, placed_top + model.tip_below + model.margin))
        z_safe_rise = max_reachable_z(cur.x, cur.y, target_z=target_rise, min_z=cur.z + 0.005, step=0.002)
        if z_safe_rise is not None and z_safe_rise > cur.z + 0.005:
            steps += _vertical(cur, z_safe_rise, step, speed, min_t, gap_release, world_after, model, None, "rise", ignore_board=True)
            cur = Pose(cur.x, cur.y, z_safe_rise)
        # Fold up to high elevation at the socket's azimuth, then rotate home in the air
        theta_sock = math.atan2(cur.y, cur.x - 0.012)
        q_high_sock = [theta_sock, home[1], home[2], home[3]]
        q_cur = ik(cur)
        steps.append({"type": "move", "name": "fold_up",
                      "points": _checked_joint_move(q_cur, q_high_sock, world_after, model, gap_release, what="cur -> high")})
        t_home = max(0.5, abs(theta_sock - home[0]) / 0.8)
        steps.append({"type": "move", "name": "home", "points": [(home, t_home)]})

    info = {"yaw_error_deg": math.degrees(predicted_socket_error(peg, sock)), "open_gap": gap_open,
            "peg_height": peg_h, "grasp_depth": d, "carry_z": h["z_carry"],
            "route": "over" if h["over"] else "around", "warnings": warnings}
    return steps, info


def plan_order(pairs, cfg, sockets, start_q=None, preferred=None):
    """Find an ORDER in which every peg can be placed without collisions.

    The order matters: a peg already standing in its socket is an obstacle for the
    next one, and a peg still on the table may block a path. Depth-first search over
    orders (max 4! = 24, usually the first try works). Returns (order, errors);
    `order` is the longest feasible sequence found (all pairs if everything works).
    """
    preferred = preferred or []
    pairs = sorted(pairs, key=lambda ps: preferred.index(ps[0]["shape"]) if ps[0]["shape"] in preferred else 99)
    best, errors = [], {}

    def dfs(done, left):
        nonlocal best
        if len(done) > len(best):
            best = list(done)
        if not left:
            return True
        for i, (peg, sock) in enumerate(left):
            others = [p for p, _ in left if p is not peg]
            try:
                plan_pick_place(peg, sock, cfg, start_q, obstacles=others, placed=done, sockets=sockets)
            except PlanError as e:
                errors[peg["color"]] = str(e)
                continue
            if dfs(done + [(peg, sock)], left[:i] + left[i + 1:]):
                return True
        return False

    dfs([], list(pairs))
    return best, errors