"""Glue: one RGB-D frame -> list of pegs and sockets in the robot base frame,
and fusion of the two cameras into one scene.

Why the extra "bird's-eye" step?
  A camera looking at the table at an ANGLE sees a square as a trapezium,
  so counting corners in the raw image fails. We therefore:
    1. turn every masked pixel into a 3-D point in the robot base frame,
    2. keep the highest points (= the peg's top face),
    3. project the outline onto that horizontal plane (metres),
    4. classify the shape on this true top view.
  This works for top-down AND angled cameras.

Pure Python + OpenCV (no ROS) -> test offline with scripts/offline_vision_test.py.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .color_segmentation import segment_pegs
from .shape_detection import SYMMETRY, classify_shape, polygon_yaw
from .socket_detection import find_holes, ring_mask

MM = 1000.0   # metric contours are drawn on a 1 px = 1 mm canvas


def pixels_to_base(us, vs, depth_m, K, T):
    """Pixels with their own depth -> base-frame points (N,3); drops invalid depth."""
    z = depth_m[vs, us]
    ok = np.isfinite(z) & (z > 0.05)
    us, vs, z = us[ok], vs[ok], z[ok]
    Kinv = np.linalg.inv(np.asarray(K, float))
    rays = np.stack([us, vs, np.ones_like(us)], 0).astype(float)
    pc = (Kinv @ rays) * z
    T = np.asarray(T, float)
    return (T[:3, :3] @ pc + T[:3, 3:4]).T, us, vs


def rays_to_plane(uv, z_plane, K, T):
    """Intersect the camera rays through pixels `uv` (N,2) with the plane z = z_plane."""
    T = np.asarray(T, float)
    Kinv = np.linalg.inv(np.asarray(K, float))
    d = T[:3, :3] @ (Kinv @ np.c_[uv, np.ones(len(uv))].T)          # ray directions in base
    o = T[:3, 3]
    t = (z_plane - o[2]) / d[2]
    return (o[:, None] + d * t).T                                     # (N,3)


def metric_shape(contour_px, z_plane, K, T, eps, min_area_mm=750.0):
    """Classify a pixel outline AFTER projecting it onto the horizontal plane z_plane."""
    pts = rays_to_plane(contour_px.reshape(-1, 2).astype(float), z_plane, K, T)[:, :2]
    if len(pts) < 3:
        return None
    c_mm = np.round(pts * MM).astype(np.int32).reshape(-1, 1, 2)
    if cv2.contourArea(c_mm) < min_area_mm:
        return None
    return _classify_mm(c_mm, eps)


def top_face(mask, depth_m, K, T, band=0.008, min_z=None, max_z=None):
    """3-D points of the peg's TOP face (highest points in base frame) and its height."""
    vs, us = np.nonzero(mask)
    if us.size < 30:
        return None
    pts, us, vs = pixels_to_base(us, vs, depth_m, K, T)
    if len(pts) < 30:
        return None

    # Filter out points that are physically impossible for a peg on the table
    # (e.g. robot arm links/fingers above the peg, or reflections)
    if min_z is not None:
        valid = pts[:, 2] >= min_z
        pts, us, vs = pts[valid], us[valid], vs[valid]
    if max_z is not None:
        valid = pts[:, 2] <= max_z
        pts, us, vs = pts[valid], us[valid], vs[valid]
    if len(pts) < 30:
        return None

    # robust "highest surface": the highest 2 mm height-bin that holds enough
    # points (a few noisy "flying pixels" above the peg are ignored)
    z = pts[:, 2]
    bins = np.arange(z.min(), z.max() + 0.004, 0.002)
    hist, edges = np.histogram(z, bins)
    full = np.nonzero(hist >= max(15, 0.03 * len(z)))[0]
    z_top = float(edges[full[-1] + 1]) if len(full) else float(np.percentile(z, 98))
    # keep a thin slab around the top; also drops "flying pixels" above it
    sel = (pts[:, 2] > z_top - band) & (pts[:, 2] < z_top + band / 2)
    if sel.sum() < 30:
        return None
    return pts[sel], float(np.median(pts[sel, 2]))


def _classify_mm(c_mm, eps):
    """Classify an outline given in integer millimetres; returns detection dict."""
    m = cv2.moments(c_mm)
    if m["m00"] == 0:
        return None
    cx, cy = m["m10"] / m["m00"] / MM, m["m01"] / m["m00"] / MM
    shape, verts, conf = classify_shape(c_mm, eps)
    yaw = polygon_yaw(verts / MM, (cx, cy), SYMMETRY[shape])
    pts = c_mm.reshape(-1, 2) / MM
    size = float(2 * np.max(np.linalg.norm(pts - (cx, cy), axis=1)))
    return dict(shape=shape, conf=round(conf, 3), x=float(cx), y=float(cy), yaw=float(yaw), size=size)


def footprint_shape(points_xy, eps, close_mm=5, open_mm=3):
    """Top view of a 3-D point set: draw the points on a 1 mm grid -> outline -> classify.

    Uses each point's TRUE x-y, so pixels on the peg's side walls land ON the
    peg's outline instead of outside it (unlike projecting the image outline).
    """
    if len(points_xy) < 30:
        return None
    mm = np.round(points_xy * MM).astype(np.int32)
    lo = mm.min(0) - 10
    g = mm - lo
    canvas = np.zeros((g[:, 1].max() + 11, g[:, 0].max() + 11), np.uint8)
    canvas[g[:, 1], g[:, 0]] = 255
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((close_mm, close_mm), np.uint8))
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, np.ones((open_mm, open_mm), np.uint8))   # cut thin spikes
    cs, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea) + lo.reshape(1, 1, 2)
    return _classify_mm(c.astype(np.int32), eps)


def process_frame(bgr, depth_m, K, T_base_cam, colors_cfg, task_cfg, cam_name="cam"):
    """Detect pegs (by colour) and sockets (holes in the board) in one camera."""
    vis = task_cfg.get("vision", {})
    eps = vis.get("approx_eps_frac", 0.03)
    T = np.asarray(T_base_cam, float)
    view = float(max(0.05, -T[2, 2]))   # 1.0 = camera looks straight down (most trustworthy)
    result = {"pegs": [], "sockets": []}
    if depth_m is None:
        return result

    # ---------- pegs ----------
    expected_table_z = float(task_cfg.get("table_z", 0.0))
    expected_peg_h = float(task_cfg.get("peg_height", 0.098))
    expected_z_top = expected_table_z + expected_peg_h
    min_z = expected_table_z + 0.01
    max_z = expected_z_top + 0.035

    peg_masks = segment_pegs(bgr, colors_cfg, vis.get("min_peg_area_px", 300))
    for color, (contour, mask) in peg_masks.items():
        top = top_face(mask, depth_m, K, T, vis.get("top_band_m", 0.02), min_z=min_z, max_z=max_z)
        det = None
        if top is not None:
            top_pts, z_top = top
            det = footprint_shape(top_pts[:, :2], eps)
        
        # Fallback if depth was missing (e.g. dark blue absorbing IR) or corrupted by arm occlusion:
        # project the top portion of the color mask onto the known peg top height plane
        if det is None:
            vs, us = np.nonzero(mask)
            if len(vs) >= 30:
                v_min, v_max = vs.min(), vs.max()
                h_top = max(25, int(0.50 * (v_max - v_min)))
                sel = vs <= (v_min + h_top)
                top_uv = np.c_[us[sel], vs[sel]].astype(float)
                top_pts_base = rays_to_plane(top_uv, expected_z_top, K, T)
                det = footprint_shape(top_pts_base[:, :2], eps)
                z_top = expected_z_top

        if det is None:
            continue

        if not task_cfg.get("use_depth_height", True) or z_top < 0.04:
            z_top = expected_z_top
        fixed = colors_cfg["pegs"][color].get("shape")
        det.update(color=color, seen_shape=det["shape"], shape=fixed or det["shape"], z_top=z_top,
                   cam=cam_name, view=view, contour=contour)
        result["pegs"].append(det)

    # ---------- sockets ----------
    board_ranges = colors_cfg.get("board", {}).get("hsv")
    if board_ranges:
        peg_cnts = [cnt for cnt, _ in peg_masks.values()]
        board_mask, holes = find_holes(
            bgr, board_ranges, vis.get("min_socket_area_px", 200), peg_contours=peg_cnts
        )
        for h in holes:
            ring = cv2.bitwise_and(ring_mask(board_mask.shape, h), board_mask)
            vs, us = np.nonzero(ring)
            pts, _, _ = pixels_to_base(us, vs, depth_m, K, T)
            expected_board_z = expected_table_z + float(task_cfg.get("place", {}).get("board_height", 0.008))
            if len(pts) >= 10:
                z_board = float(np.median(pts[:, 2]))
            else:
                z_board = expected_board_z
            # The board is a flat cardboard strip on the table (< 15 mm thick).
            # If ring touched an adjacent tall peg (e.g. z > 25 mm), clamp back to expected board height.
            if z_board > (expected_table_z + 0.025) or z_board < (expected_table_z - 0.010):
                z_board = expected_board_z
            det = metric_shape(h, z_board, K, T, eps)
            if det is None:
                continue

            # Ignore holes that are already occupied by a peg (e.g. after a peg is placed).
            # A peg sitting in a hole distorts the contour and causes false shape classification.
            # Center-to-center distance threshold is 20 mm (adjacent sockets are ~30 mm apart).
            occupied = False
            for p in result["pegs"]:
                if math.hypot(det["x"] - p["x"], det["y"] - p["y"]) < 0.020:
                    occupied = True
                    break
                pc = p.get("contour")
                if pc is not None:
                    pm = cv2.moments(pc)
                    if pm["m00"] > 0:
                        pu, pv = float(pm["m10"] / pm["m00"]), float(pm["m01"] / pm["m00"])
                        if cv2.pointPolygonTest(h, (pu, pv), False) >= 0:
                            occupied = True
                            break
            if occupied:
                continue

            det.update(z_top=z_board, cam=cam_name, view=view, contour=h)
            result["sockets"].append(det)
    return result


def _avg_angle(yaws, weights, n):
    if n == 0 or not yaws:
        return 0.0
    s = sum(w * math.sin(n * a) for a, w in zip(yaws, weights))
    c = sum(w * math.cos(n * a) for a, w in zip(yaws, weights))
    return (math.atan2(s, c) / n) % (2 * math.pi / n)


def _merge(items):
    # weight = shape confidence x how straight-down the camera looks
    w = [max(i["conf"], 0.05) * i.get("view", 1.0) ** 2 for i in items]
    out = dict(items[0])
    # if the cameras disagree on the shape, trust the higher weight
    out["shape"] = items[int(np.argmax(w))]["shape"]
    for k in ("x", "y", "z_top", "size"):
        out[k] = float(np.average([i[k] for i in items], weights=w))
    same = [(i["yaw"], wi) for i, wi in zip(items, w) if i["shape"] == out["shape"]]
    out["yaw"] = _avg_angle([a for a, _ in same], [b for _, b in same], SYMMETRY[out["shape"]])
    out["conf"] = float(max(i["conf"] for i in items))
    out["cam"] = "+".join(sorted({i["cam"] for i in items}))
    out.pop("contour", None)
    return out


def fuse(results, merge_dist=0.015):
    """Combine detections from both cameras.

    Pegs: same colour = same peg.  Sockets: closer than `merge_dist` m = same
    socket. Values are confidence-weighted means.
    """
    pegs_by_color = {}
    for r in results:
        for p in r["pegs"]:
            pegs_by_color.setdefault(p["color"], []).append(p)
    pegs = [_merge(v) for v in pegs_by_color.values()]

    groups = []
    for r in results:
        for s in r["sockets"]:
            for g in groups:
                if math.hypot(g[0]["x"] - s["x"], g[0]["y"] - s["y"]) < merge_dist:
                    g.append(s)
                    break
            else:
                groups.append([s])
    merged_sockets = [_merge(g) for g in groups]
    # Filter out any socket candidate that coincides with a peg (occupied socket)
    avail_sockets = [
        s for s in merged_sockets
        if not any(math.hypot(s["x"] - p["x"], s["y"] - p["y"]) < 0.020 for p in pegs)
    ]
    return {"pegs": pegs, "sockets": avail_sockets}


def draw(bgr, result, K, T_base_cam):
    """Debug overlay: draw high-contrast contour boundaries and 3D labels."""
    img = bgr.copy()
    T_cam_base = np.linalg.inv(np.asarray(T_base_cam))
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]

    COLOR_MAP = {
        "orange": (0, 140, 255),    # BGR orange
        "red":    (0, 140, 255),    # BGR orange
        "green":  (0, 255, 0),      # BGR green
        "blue":   (255, 120, 0),    # BGR sky blue
        "yellow": (0, 255, 255),    # BGR yellow
    }

    # 1. Draw Pegs with their individual colors and boundaries
    for p in result.get("pegs", []):
        colr = COLOR_MAP.get(p.get("color"), (0, 255, 0))
        if "contour" in p and p["contour"] is not None:
            cv2.drawContours(img, [p["contour"]], -1, colr, 2)
        
        pt = T_cam_base @ np.array([p["x"], p["y"], p["z_top"], 1.0])
        if pt[2] > 0:
            u, v = int(fx * pt[0] / pt[2] + cx), int(fy * pt[1] / pt[2] + cy)
            cv2.circle(img, (u, v), 5, (0, 0, 255), -1)
            r_dist = math.hypot(p["x"] - 0.012, p["y"])
            tag = f"{p.get('color', '').upper()} {p['shape']} ({r_dist*100:.1f}cm)"
            cv2.putText(img, tag, (u - 40, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, tag, (u - 40, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colr, 1, cv2.LINE_AA)

    # 2. Draw Sockets with magenta/cyan outlines and reachability warnings
    for s in result.get("sockets", []):
        r_dist = math.hypot(s["x"] - 0.012, s["y"])
        is_reachable = (r_dist <= 0.205)
        # Cyan if reachable, bright red if too far
        colr = (255, 255, 0) if is_reachable else (0, 0, 255)
        
        if "contour" in s and s["contour"] is not None:
            cv2.drawContours(img, [s["contour"]], -1, colr, 2)
        
        pt = T_cam_base @ np.array([s["x"], s["y"], s["z_top"], 1.0])
        if pt[2] > 0:
            u, v = int(fx * pt[0] / pt[2] + cx), int(fy * pt[1] / pt[2] + cy)
            cv2.circle(img, (u, v), 5, colr, -1)
            status = "OK" if is_reachable else "TOO FAR (move closer)"
            tag = f"SOCKET: {s['shape']} ({r_dist*100:.1f}cm - {status})"
            cv2.putText(img, tag, (u - 50, v + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, tag, (u - 50, v + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colr, 1, cv2.LINE_AA)

    return img
