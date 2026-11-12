"""Find the centre of each socket on the MDF plate and save it for trajectory planning.

    python3 locate_sockets.py           # one iteration (30 frames), then exit
    python3 locate_sockets.py --live    # keep running; README.md rewritten every iteration

Every iteration rewrites, next to this script:
    README.md    socket centres in camera, marker and robot frames + CSV block
    sockets.png  annotated camera image

Method, per frame:
  1. Socket = patch of teal foam whose border is mostly the painted rim colour.
     Its shape (triangle/square/hexagon/circle) comes from comparing the
     outline's radius-vs-angle profile with each ideal shape.
  2. 3D centre = the hole's centroid pixel, placed at the depth of the plate
     surface just around the hole, deprojected with the camera intrinsics.
  3. Marker frame = ArUco marker centre; x along its top edge, y up the marker,
     z out of the table. The table plane is fitted from depth around the
     marker, so the marker's printed size is not needed.
The median over all frames of an iteration is saved.
"""
import argparse
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

HERE = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(HERE, "README.md")
IMAGE = os.path.join(HERE, "sockets.png")

# ---- Settings ------------------------------------------------------------
# HSV ranges (OpenCV: H 0-179), tuned on the RealSense feed.
RIM_HSV = ((0, 90, 80), (15, 255, 255))      # painted rim around each cut-out
FOAM_HSV = ((75, 100, 60), (100, 255, 255))  # teal foam seen through an empty socket

MARKER_ID = None                  # None = first 4x4 ArUco marker found
# Robot base in the marker frame. PLACEHOLDER (from peg_detecter_robot_frame.py):
# base 4 cm along the marker's x axis, robot axes parallel to the marker's. Measure it.
BASE_XYZ_M = np.array([0.040, 0.0, 0.0])
BASE_YAW_DEG = 0.0

FRAMES_PER_ITERATION = 30
SHAPES = ("triangle", "square", "hexagon", "circle")
SYMMETRY_DEG = {"triangle": 120.0, "square": 90.0, "hexagon": 60.0, "circle": None}

# ---- Shape classification --------------------------------------------------
N_BINS, SMOOTH, SIGMA = 360, 7, 0.03
GRID = np.arange(N_BINS) * 2 * np.pi / N_BINS


def smooth(p):
    pad = SMOOTH // 2
    return np.convolve(np.concatenate([p[-pad:], p, p[:pad]]), np.ones(SMOOTH) / SMOOTH, mode="valid")


def template(shape):
    """Ideal radius-vs-angle of a regular polygon with a corner at angle 0."""
    if SYMMETRY_DEG[shape] is None:
        return np.ones(N_BINS)
    sector = np.radians(SYMMETRY_DEG[shape])
    p = smooth(np.cos(sector / 2) / np.cos((GRID % sector) - sector / 2))
    return p / p.mean()


TEMPLATES = {s: template(s) for s in SHAPES}


def classify(contour):
    """(shape, confidence, centroid_px) of a closed outline, or None."""
    m = cv2.moments(cv2.convexHull(contour))
    if m["m00"] == 0:
        return None
    c = (m["m10"] / m["m00"], m["m01"] / m["m00"])
    pts = contour.reshape(-1, 2).astype(float) - c
    theta = np.arctan2(pts[:, 1], pts[:, 0]) % (2 * np.pi)
    order = np.argsort(theta)
    prof = smooth(np.interp(GRID, theta[order], np.hypot(*pts.T)[order], period=2 * np.pi))
    prof /= prof.mean()

    err = {}
    for s in SHAPES:
        if SYMMETRY_DEG[s] is None:
            err[s] = np.sqrt(np.mean((prof - 1) ** 2))
            continue
        n = round(360 / SYMMETRY_DEG[s])
        yaw = np.angle(np.sum((prof - 1) * np.exp(1j * n * GRID))) / n  # n-th harmonic -> corner direction
        shifted = np.roll(TEMPLATES[s], int(round(yaw / (2 * np.pi) * N_BINS)))
        err[s] = np.sqrt(np.mean((prof - shifted) ** 2))
    best = min(err.values())
    w = {s: np.exp(-(e * e - best * best) / SIGMA ** 2) for s, e in err.items()}
    shape = max(w, key=w.get)
    return shape, w[shape] / sum(w.values()), c


# ---- Sockets in 3D ---------------------------------------------------------
def deproject(intr, u, v, z):
    return np.array([(u - intr.ppx) / intr.fx * z, (v - intr.ppy) / intr.fy * z, z])


def find_sockets(bgr, depth_m, intr):
    """{shape: (centroid_px, xyz_camera, contour)}, one per shape (most confident wins)."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    rim = cv2.inRange(hsv, np.array(RIM_HSV[0]), np.array(RIM_HSV[1])) > 0
    foam = cv2.morphologyEx(cv2.inRange(hsv, np.array(FOAM_HSV[0]), np.array(FOAM_HSV[1])),
                            cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foam)
    min_area = 2e-4 * bgr.shape[0] * bgr.shape[1]
    best = {}
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        blob = (labels == i).astype(np.uint8) * 255
        border = (cv2.dilate(blob, np.ones((9, 9), np.uint8)) > 0) & (blob == 0)
        if np.count_nonzero(rim & border) < 0.5 * np.count_nonzero(border):
            continue  # not ringed by rim paint: the foam sheet itself, or clutter
        contour = max(cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0],
                      key=cv2.contourArea)
        res = classify(contour)
        if res is None:
            continue
        shape, conf, c = res
        # Depth of the plate surface: a ring 3-15 px outside the hole.
        ring = ((cv2.dilate(blob, np.ones((31, 31), np.uint8)) > 0)
                & ~(cv2.dilate(blob, np.ones((7, 7), np.uint8)) > 0) & (depth_m > 0))
        if np.count_nonzero(ring) < 20:
            continue
        xyz = deproject(intr, c[0], c[1], float(np.median(depth_m[ring])))
        if shape not in best or conf > best[shape][3]:
            best[shape] = (np.array(c), xyz, contour, conf)
    return {s: v[:3] for s, v in best.items()}


# ---- Marker frame ----------------------------------------------------------
def find_marker(bgr):
    """Corners (TL, TR, BR, BL) of the ArUco marker, or None."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    if hasattr(cv2.aruco, "ArucoDetector"):  # OpenCV >= 4.7
        corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=cv2.aruco.DetectorParameters_create())
    if ids is None:
        return None
    for c, i in zip(corners, ids.flatten()):
        if MARKER_ID is None or int(i) == MARKER_ID:
            return c.reshape(4, 2).astype(float)
    return None


def marker_pose(corners, depth_m, intr):
    """(R, origin) in the camera frame; columns of R are the marker x, y, z axes."""
    # Fit the table plane starting from the marker's pixels, then widen to 3x its
    # size keeping only points on that plane (so the arm or a peg is never fitted).
    mid = corners.mean(axis=0)
    marker = np.zeros(depth_m.shape, np.uint8)
    cv2.fillConvexPoly(marker, corners.astype(np.int32), 255)
    region = np.zeros(depth_m.shape, np.uint8)
    cv2.fillConvexPoly(region, np.round(mid + 3 * (corners - mid)).astype(np.int32), 255)
    vs, us = np.nonzero((region > 0) & (depth_m > 0))
    keep = marker[vs, us] > 0
    if keep.sum() < 50:
        return None
    z = depth_m[vs, us]
    pts = np.column_stack([(us - intr.ppx) / intr.fx * z, (vs - intr.ppy) / intr.fy * z, z])
    for _ in range(4):
        centre = pts[keep].mean(axis=0)
        normal = np.linalg.svd(pts[keep] - centre)[2][2]
        new_keep = np.abs((pts - centre) @ normal) < 0.004
        if new_keep.sum() < 50:
            break
        keep = new_keep
    if normal[2] > 0:
        normal = -normal  # toward the camera
    rays = np.column_stack([(corners[:, 0] - intr.ppx) / intr.fx, (corners[:, 1] - intr.ppy) / intr.fy, np.ones(4)])
    tl, tr, br, bl = rays * ((normal @ centre) / (rays @ normal))[:, None]  # corners on the plane
    x = (tr - tl) + (br - bl)
    x /= np.linalg.norm(x)
    zax = np.cross(x, (tl - bl) + (tr - br))
    zax /= np.linalg.norm(zax)
    return np.column_stack([x, np.cross(zax, x), zax]), (tl + tr + br + bl) / 4


def average_pose(poses):
    origin = np.median([o for _, o in poses], axis=0)
    u, _, vt = np.linalg.svd(np.mean([R for R, _ in poses], axis=0))
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return u @ vt, origin


def to_robot(p_marker):
    a = np.radians(BASE_YAW_DEG)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return Rz.T @ (p_marker - BASE_XYZ_M)


# ---- One iteration -----------------------------------------------------------
def summarise(samples, poses, n_frames):
    pose = average_pose(poses) if poses else None
    print(f"== {n_frames} frames, marker seen in {len(poses)}")
    if pose is None:
        print("   WARN marker not found (arm or hand over it?): camera frame only")
    result = {}
    for shape in SHAPES:
        got = samples[shape]
        if not got:
            print(f"   {shape:8s} NOT FOUND")
            continue
        cams = np.array([g[1] for g in got])
        cam = np.median(cams, axis=0)
        mk = pose[0].T @ (cam - pose[1]) if pose is not None else None
        result[shape] = {
            "seen": len(got),
            "pixel": np.median([g[0] for g in got], axis=0),
            "camera": cam,
            "marker": mk,
            "robot": to_robot(mk) if mk is not None else None,
            "jitter_mm": cams.std(axis=0).max() * 1000,
        }
        where = f"marker ({mk[0]*100:6.1f}, {mk[1]*100:6.1f}, {mk[2]*100:5.1f}) cm" if mk is not None else ""
        print(f"   {shape:8s} {len(got):3d}/{n_frames}  {where}  jitter {result[shape]['jitter_mm']:.1f} mm")
    return result


def layout(result):
    """Order, spacing and straightness of the sockets (camera-only check)."""
    if len(result) < 3:
        return None
    frame = "marker" if all(r["marker"] is not None for r in result.values()) else "camera"
    names = list(result)
    pts = np.array([result[n][frame] for n in names])
    mid = pts.mean(axis=0)
    axis = np.linalg.svd(pts - mid)[2][0]
    order = np.argsort((pts - mid) @ axis)
    names, pts = [names[i] for i in order], pts[order]
    gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    off = np.linalg.norm(np.cross(pts - mid, axis), axis=1).max()
    print(f"   layout: {' -> '.join(names)}, spacing {', '.join(f'{g*100:.1f}' for g in gaps)} cm, "
          f"off-line {off*1000:.1f} mm")
    return frame, names, gaps, off


def write_readme(result, lay, n_frames, n_marker, iteration):
    def v3(p):
        return "n/a" if p is None else f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})"

    def c3(p):
        return ",," if p is None else f"{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}"

    L = [
        "# Socket Centroids",
        "",
        f"**Saved:** {time.strftime('%Y-%m-%d %H:%M:%S')} (iteration {iteration}, rewritten every iteration)  ",
        f"**Frames averaged:** {n_frames} (median), ArUco marker seen in {n_marker}  ",
        "**Point:** centre of the socket opening, at the plate's top surface. Units: metres.",
        "",
        "## Frames",
        "",
        "| Frame | Origin | Axes |",
        "| :--- | :--- | :--- |",
        "| Camera | RealSense colour lens | x right, y down, z forward |",
        "| Marker | ArUco marker centre | x along marker top edge, y up the marker, z up out of the table |",
        f"| Robot | robot base | base at {v3(BASE_XYZ_M)} m, yaw {BASE_YAW_DEG:.1f} deg in the marker frame "
        "(**placeholder: measure before use**) |",
        "",
        "## Socket centres",
        "",
        "| Socket | Seen | Pixel (u, v) | Robot (X, Y, Z) | Marker (X, Y, Z) | Camera (X, Y, Z) |",
        "| :--- | ---: | :--- | :--- | :--- | :--- |",
    ]
    for s, r in result.items():
        L.append(f"| **{s}** | {r['seen']}/{n_frames} | ({r['pixel'][0]:.1f}, {r['pixel'][1]:.1f}) | "
                 f"`{v3(r['robot'])}` | `{v3(r['marker'])}` | `{v3(r['camera'])}` |")
    missing = [s for s in SHAPES if s not in result]
    if missing:
        L += ["", f"**Not found:** {', '.join(missing)}"]
    if lay:
        frame, names, gaps, off = lay
        L += ["", f"**Layout ({frame} frame):** {' -> '.join(names)}; spacing "
              f"{', '.join(f'{g*100:.1f}' for g in gaps)} cm; max off-line {off*1000:.1f} mm"]
    L += ["", "## CSV (for the trajectory planner)", "", "```csv",
          "socket,robot_x_m,robot_y_m,robot_z_m,marker_x_m,marker_y_m,marker_z_m,cam_x_m,cam_y_m,cam_z_m"]
    L += [f"{s},{c3(r['robot'])},{c3(r['marker'])},{c3(r['camera'])}" for s, r in result.items()]
    L += ["```", ""]
    with open(README, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def annotate(bgr, found, corners, status=None):
    img = bgr.copy()
    if corners is not None:
        cv2.polylines(img, [corners.astype(np.int32)], True, (255, 0, 255), 2)
    for shape, (px, _, contour) in found.items():
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
        cv2.drawMarker(img, tuple(int(v) for v in px), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.putText(img, shape, (int(px[0]) + 8, int(px[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if status:
        cv2.putText(img, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return img


# ---- Main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="keep running; rewrite README.md every iteration (q quits)")
    ap.add_argument("--frames", type=int, default=FRAMES_PER_ITERATION, help="frames averaged per iteration")
    args = ap.parse_args()

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    scale = pipe.start(cfg).get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)

    iteration, status = 0, "collecting..."
    samples, poses, n_frames = {s: [] for s in SHAPES}, [], 0
    try:
        for _ in range(30):  # let auto-exposure settle
            pipe.wait_for_frames()
        while True:
            f = align.process(pipe.wait_for_frames())
            d, c = f.get_depth_frame(), f.get_color_frame()
            if not d or not c:
                continue
            intr = c.profile.as_video_stream_profile().intrinsics
            bgr = np.asanyarray(c.get_data()).copy()
            depth_m = np.asanyarray(d.get_data()).astype(np.float32) * scale

            corners = find_marker(bgr)
            pose = marker_pose(corners, depth_m, intr) if corners is not None else None
            found = find_sockets(bgr, depth_m, intr)
            n_frames += 1
            if pose is not None:
                poses.append(pose)
            for s, v in found.items():
                samples[s].append(v)

            if n_frames >= args.frames:
                iteration += 1
                print(f"\n#### iteration {iteration}")
                result = summarise(samples, poses, n_frames)
                write_readme(result, layout(result), n_frames, len(poses), iteration)
                status = f"iteration {iteration} saved {time.strftime('%H:%M:%S')}: {len(result)}/4 sockets"
                cv2.imwrite(IMAGE, annotate(bgr, found, corners))
                print(f"   saved {os.path.relpath(README)} and {os.path.relpath(IMAGE)}")
                samples, poses, n_frames = {s: [] for s in SHAPES}, [], 0
                if not args.live:
                    break

            if args.live:
                cv2.imshow("socket centroids (q to quit)", annotate(bgr, found, corners, status))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
