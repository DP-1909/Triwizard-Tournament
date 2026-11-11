import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs

# =====================================================================
# 1. WORKSPACE & PEG PHYSICAL PARAMETERS (in meters)
# =====================================================================
# 4 cm (0.040 m) offset from the 4x4 marker center to the robot base center
BASE_OFFSET_X_M = 0.040
BASE_OFFSET_Y_M = 0.000
BASE_OFFSET_Z_M = 0.000

# Fixed 10 cm (0.100 m) peg height
PEG_HEIGHT_M = 0.100

# Output file configuration
README_FILENAME = "README.md"
EXPECTED_PEG_COUNT = 4
WARMUP_FRAMES = 30
AUTO_EXIT_ON_SAVE = False

# =====================================================================
# 2. TUNED HSV THRESHOLDS & KNOWN COLOR-TO-SHAPE MAPPING
# =====================================================================
PEG_CONFIG = {
    "Blue_Peg": {
        "shape": "Square",
        "lower": np.array([94, 180, 100]),
        "upper": np.array([115, 255, 255]),
    },
    # Updated for Clicked Pixel (496, 68) -> H: 27, S: 206, V: 254
    # Lowered S_min to 100 and V_min to 130 so bright top-surface glare isn't clipped
    "Yellow_Peg": {
        "shape": "Triangle",
        "lower": np.array([20, 100, 130]),
        "upper": np.array([33, 255, 255]),
    },
    "Orange_Peg": {
        "shape": "Hexagon",
        "lower": np.array([3, 120, 150]),
        "upper": np.array([18, 255, 255]),
    },
    "Green_Peg": {
        "shape": "Circle",
        "lower": np.array([34, 70, 115]),
        "upper": np.array([55, 200, 255]),
    },
}

MIN_CONTOUR_AREA = 250
MAX_CONTOUR_AREA = 20000


def detect_4x4_marker(gray_frame):
    """Detects the 4x4 fiducial marker across OpenCV versions."""
    candidate_dicts = [
        cv2.aruco.DICT_4X4_250,
        cv2.aruco.DICT_APRILTAG_16h5,
    ]

    for dict_id in candidate_dicts:
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(gray_frame)
        else:
            aruco_dict = cv2.aruco.Dictionary_get(dict_id)
            params = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray_frame, aruco_dict, parameters=params
            )

        if ids is not None and len(ids) > 0:
            marker_corners = np.array(corners[0], dtype=np.float32).reshape(4, 2)
            marker_id = int(np.array(ids).flatten()[0])
            return marker_corners, marker_id

    return None, None


def get_robust_depth(depth_image, depth_scale, cx, cy, kernel_size=5):
    """Returns median non-zero depth (in meters) around pixel (cx, cy)."""
    h, w = depth_image.shape
    half_k = kernel_size // 2
    x_min, x_max = max(0, cx - half_k), min(w, cx + half_k + 1)
    y_min, y_max = max(0, cy - half_k), min(h, cy + half_k + 1)

    patch = depth_image[y_min:y_max, x_min:x_max]
    valid_pixels = patch[patch > 0]

    if len(valid_pixels) == 0:
        return 0.0

    return float(np.median(valid_pixels) * depth_scale)


def isolate_peg_top_face(
    contour, mask_shape, intrinsics, depth_image, depth_scale, board_depth_m
):
    """Isolates the top face of a 10 cm tall peg and uses its Convex Hull

    area moments to compute a glare-resistant geometric centroid.
    """
    cnt_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(cnt_mask, [contour], -1, 255, -1)

    # 1. Depth-plane pre-filter around the expected 10 cm top surface
    if board_depth_m is not None and board_depth_m > PEG_HEIGHT_M + 0.10:
        expected_top_d = board_depth_m - PEG_HEIGHT_M
        depth_m_img = depth_image * depth_scale
        top_depth_mask = (
            (depth_m_img >= expected_top_d - 0.025)
            & (depth_m_img <= expected_top_d + 0.025)
        ).astype(np.uint8) * 255

        depth_filtered = cv2.bitwise_and(cnt_mask, top_depth_mask)
        if np.count_nonzero(depth_filtered) >= 45:
            cnt_mask = depth_filtered

    ys, xs = np.nonzero(cnt_mask)
    if len(xs) < 10:
        hull = cv2.convexHull(contour)
        M = cv2.moments(hull)
        cx = int(M["m10"] / max(M["m00"], 1e-5))
        cy = int(M["m01"] / max(M["m00"], 1e-5))
        return cx, cy, hull

    mean_x, mean_y = float(np.mean(xs)), float(np.mean(ys))
    vx = mean_x - intrinsics.ppx
    vy = mean_y - intrinsics.ppy
    norm = np.hypot(vx, vy)

    if norm < 5.0:
        hull = cv2.convexHull(contour)
        M = cv2.moments(hull)
        return (
            int(M["m10"] / max(M["m00"], 1e-5)),
            int(M["m01"] / max(M["m00"], 1e-5)),
            hull,
        )

    # Radial unit vector r_hat (from camera principal point to peg) and transverse t_hat
    rx, ry = vx / norm, vy / norm
    tx, ty = -ry, rx

    r_proj = (xs - intrinsics.ppx) * rx + (ys - intrinsics.ppy) * ry
    t_proj = (xs - intrinsics.ppx) * tx + (ys - intrinsics.ppy) * ty

    w_transverse = float(
        np.percentile(t_proj, 98) - np.percentile(t_proj, 2)
    )
    r_max = float(np.percentile(r_proj, 99))
    r_min = float(np.percentile(r_proj, 1))
    radial_length = r_max - r_min

    # 2. Slice off the inner vertical side wall
    if radial_length > w_transverse * 1.05:
        if board_depth_m is not None and board_depth_m > PEG_HEIGHT_M + 0.10:
            scale_inv = board_depth_m / (board_depth_m - PEG_HEIGHT_M)
            r_cutoff_proj = r_min * scale_inv
            r_cutoff = float(
                np.clip(
                    r_cutoff_proj,
                    r_max - 1.25 * w_transverse,
                    r_max - 0.65 * w_transverse,
                )
            )
        else:
            r_cutoff = r_max - w_transverse

        keep_idx = r_proj >= r_cutoff
        if np.count_nonzero(keep_idx) >= 15:
            top_xs = xs[keep_idx]
            top_ys = ys[keep_idx]

            top_pts = np.column_stack((top_xs, top_ys)).astype(np.int32)
            top_hull = cv2.convexHull(top_pts)

            M = cv2.moments(top_hull)
            if M["m00"] > 0:
                top_cx = int(M["m10"] / M["m00"])
                top_cy = int(M["m01"] / M["m00"])
            else:
                top_cx = int(np.mean(top_xs))
                top_cy = int(np.mean(top_ys))

            return top_cx, top_cy, top_hull

    hull = cv2.convexHull(contour)
    M = cv2.moments(hull)
    if M["m00"] > 0:
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), hull
    return int(mean_x), int(mean_y), contour


def get_peg_top_depth(
    depth_image, depth_scale, top_cx, top_cy, board_depth_m
):
    """Returns the depth of the 10 cm peg's top surface (Z_board - 0.100 m)."""
    measured_d = get_robust_depth(
        depth_image, depth_scale, top_cx, top_cy, kernel_size=5
    )

    if board_depth_m is not None and board_depth_m > PEG_HEIGHT_M + 0.10:
        expected_top_d = board_depth_m - PEG_HEIGHT_M
        if (
            measured_d > 0
            and abs(measured_d - expected_top_d) <= 0.015
        ):
            return measured_d
        return expected_top_d

    return measured_d


def get_robot_base_in_camera_frame(marker_cam_xyz, marker_yaw_rad):
    """Computes the 3D coordinates of the Robot Base in the Camera Frame."""
    cos_t = np.cos(marker_yaw_rad)
    sin_t = np.sin(marker_yaw_rad)

    base_cam_x = (
        marker_cam_xyz[0] + cos_t * BASE_OFFSET_X_M - sin_t * BASE_OFFSET_Y_M
    )
    base_cam_y = (
        marker_cam_xyz[1] + sin_t * BASE_OFFSET_X_M + cos_t * BASE_OFFSET_Y_M
    )
    base_cam_z = marker_cam_xyz[2] - BASE_OFFSET_Z_M

    return [base_cam_x, base_cam_y, base_cam_z]


def camera_to_robot_frame(peg_cam_xyz, marker_cam_xyz, marker_yaw_rad):
    """Transforms 3D peg coordinates from Camera Frame to Marker & Robot Frames."""
    dx_cam = peg_cam_xyz[0] - marker_cam_xyz[0]
    dy_cam = peg_cam_xyz[1] - marker_cam_xyz[1]
    dz_up = marker_cam_xyz[2] - peg_cam_xyz[2]

    cos_t = np.cos(marker_yaw_rad)
    sin_t = np.sin(marker_yaw_rad)

    x_marker = cos_t * dx_cam + sin_t * dy_cam
    y_marker = -sin_t * dx_cam + cos_t * dy_cam
    z_marker = dz_up

    x_robot = x_marker - BASE_OFFSET_X_M
    y_robot = y_marker - BASE_OFFSET_Y_M
    z_robot = z_marker - BASE_OFFSET_Z_M

    return (x_robot, y_robot, z_robot), (x_marker, y_marker, z_marker)


def write_readme_once(filepath, marker_id, marker_cam_xyz, base_cam_xyz, pegs_data):
    """Writes the robot base and initial peg coordinates into README.md once."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Initial Workspace & Peg Coordinates\n\n")
        f.write(f"**Captured Timestamp:** `{timestamp}`  \n")
        f.write(f"**Reference Marker ID:** `{marker_id}`  \n")
        f.write(f"**Fixed Peg Height (m):** `{PEG_HEIGHT_M:.4f}` (10 cm)  \n")
        f.write(
            f"**Marker-to-Robot Base Offset (m):** `dX = {BASE_OFFSET_X_M:.4f}, dY = {BASE_OFFSET_Y_M:.4f}, dZ = {BASE_OFFSET_Z_M:.4f}`\n\n"
        )

        f.write("## 1. Reference & Robot Base Coordinates (in meters)\n\n")
        f.write("| Reference Point | Robot Base Frame (X, Y, Z) | Marker Frame (X, Y, Z) | Camera Frame (X, Y, Z) |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(
            f"| **Robot Base Center** | `(0.0000, 0.0000, 0.0000)` | `({BASE_OFFSET_X_M:.4f}, {BASE_OFFSET_Y_M:.4f}, {BASE_OFFSET_Z_M:.4f})` | `({base_cam_xyz[0]:.4f}, {base_cam_xyz[1]:.4f}, {base_cam_xyz[2]:.4f})` |\n"
        )
        f.write(
            f"| **Fiducial Marker Center** | `({-BASE_OFFSET_X_M:.4f}, {-BASE_OFFSET_Y_M:.4f}, {-BASE_OFFSET_Z_M:.4f})` | `(0.0000, 0.0000, 0.0000)` | `({marker_cam_xyz[0]:.4f}, {marker_cam_xyz[1]:.4f}, {marker_cam_xyz[2]:.4f})` |\n\n"
        )

        f.write("## 2. Detected Peg Center Coordinates (in meters)\n\n")
        f.write("| Peg Color | Shape | Top-Cap Pixel (u, v) | Robot Frame Top (X, Y, Z_top) | Robot Frame Mid-Grip (X, Y, Z_mid) | Camera Frame Top (X, Y, Z) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")

        for color_label, data in pegs_data.items():
            rx, ry, rz_top = data["robot_xyz"]
            rz_mid = rz_top - (PEG_HEIGHT_M / 2.0)
            cx_m, cy_m, cz_m = data["cam_xyz"]
            u, v = data["pixel"]
            shape = data["shape"]

            f.write(
                f"| **{color_label}** | {shape} | `({u}, {v})` | `({rx:.4f}, {ry:.4f}, {rz_top:.4f})` | `({rx:.4f}, {ry:.4f}, {rz_mid:.4f})` | `({cx_m:.4f}, {cy_m:.4f}, {cz_m:.4f})` |\n"
            )

        f.write("\n## 3. Easy-Parse CSV Block (for Sorting Script)\n\n")
        f.write("```csv\n")
        f.write("object,shape,robot_x_m,robot_y_m,robot_z_top_m,robot_z_mid_m,robot_z_base_m,cam_x_m,cam_y_m,cam_z_m\n")
        f.write(
            f"Robot_Base,Origin,0.0000,0.0000,0.0000,0.0000,0.0000,{base_cam_xyz[0]:.4f},{base_cam_xyz[1]:.4f},{base_cam_xyz[2]:.4f}\n"
        )
        for color_label, data in pegs_data.items():
            rx, ry, rz_top = data["robot_xyz"]
            rz_mid = rz_top - (PEG_HEIGHT_M / 2.0)
            rz_base = rz_top - PEG_HEIGHT_M
            cx_m, cy_m, cz_m = data["cam_xyz"]
            f.write(
                f"{color_label},{data['shape']},{rx:.4f},{ry:.4f},{rz_top:.4f},{rz_mid:.4f},{rz_base:.4f},{cx_m:.4f},{cy_m:.4f},{cz_m:.4f}\n"
            )
        f.write("```\n")

    print(f"\n[SUCCESS] Coordinates written once to: {os.path.abspath(filepath)}")


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align = rs.align(rs.stream.color)

    cached_marker_xyz = None
    cached_marker_yaw = 0.0
    cached_marker_poly = None
    cached_marker_id = None

    readme_written = False
    frame_count = 0

    print("Starting camera warmup (Peg height fixed at 10 cm = 0.100 m)...")
    print(" -> Green contour = Isolated Top-Cap Hull | Yellow contour = Full Peg + Side Wall")
    print(" -> README.md will auto-save once Marker + all 4 pegs are detected (or press 's').\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            frame_count += 1
            intrinsics = (
                color_frame.profile.as_video_stream_profile().intrinsics
            )
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            display_image = color_image.copy()

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # -----------------------------------------------------------------
            # STEP 1: Detect the 4x4 Base Reference Marker
            # -----------------------------------------------------------------
            corners, marker_id = detect_4x4_marker(gray)

            if corners is not None:
                mx = int(np.mean(corners[:, 0]))
                my = int(np.mean(corners[:, 1]))

                top_vec = corners[1] - corners[0]
                marker_yaw = float(np.arctan2(top_vec[1], top_vec[0]))

                marker_depth = get_robust_depth(
                    depth_image, depth_scale, mx, my, kernel_size=7
                )
                if marker_depth > 0.0:
                    cached_marker_xyz = rs.rs2_deproject_pixel_to_point(
                        intrinsics, [mx, my], marker_depth
                    )
                    cached_marker_yaw = marker_yaw
                    cached_marker_poly = corners.astype(np.int32)
                    cached_marker_id = marker_id

                cv2.polylines(
                    display_image,
                    [corners.astype(np.int32)],
                    True,
                    (255, 0, 255),
                    2,
                )
                cv2.circle(display_image, (mx, my), 5, (255, 0, 255), -1)
                cv2.putText(
                    display_image,
                    f"Marker ID:{marker_id} (Ref Locked)",
                    (mx - 40, my - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                )

            # -----------------------------------------------------------------
            # STEP 2: Segment 10 cm Pegs, Isolate Top Cap, & Compute True XYZ
            # -----------------------------------------------------------------
            blurred = cv2.GaussianBlur(color_image, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            kernel = np.ones((5, 5), np.uint8)
            close_kernel = np.ones((7, 7), np.uint8)

            board_depth_m = (
                cached_marker_xyz[2] if cached_marker_xyz is not None else None
            )
            current_frame_pegs = {}

            for color_label, cfg in PEG_CONFIG.items():
                mask = cv2.inRange(hsv, cfg["lower"], cfg["upper"])

                if cached_marker_poly is not None:
                    x_b, y_b, w_b, h_b = cv2.boundingRect(cached_marker_poly)
                    pad = 15
                    cv2.rectangle(
                        mask,
                        (max(0, x_b - pad), max(0, y_b - pad)),
                        (
                            min(mask.shape[1], x_b + w_b + pad),
                            min(mask.shape[0], y_b + h_b + pad),
                        ),
                        0,
                        -1,
                    )

                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                valid_contours = [
                    c
                    for c in contours
                    if MIN_CONTOUR_AREA <= cv2.contourArea(c) <= MAX_CONTOUR_AREA
                ]
                if not valid_contours:
                    continue

                full_cnt = max(valid_contours, key=cv2.contourArea)
                shape_name = cfg["shape"]

                # Isolate ONLY the top face of the 10 cm peg
                top_cx, top_cy, top_cnt = isolate_peg_top_face(
                    full_cnt,
                    mask.shape,
                    intrinsics,
                    depth_image,
                    depth_scale,
                    board_depth_m,
                )

                # Deproject using the 10 cm top-surface plane (Z_board - 0.100 m)
                top_depth_m = get_peg_top_depth(
                    depth_image, depth_scale, top_cx, top_cy, board_depth_m
                )
                if top_depth_m <= 0.0:
                    continue

                peg_cam_xyz = rs.rs2_deproject_pixel_to_point(
                    intrinsics, [top_cx, top_cy], top_depth_m
                )

                # Thin yellow = full contour (with side wall), Thick green = isolated top cap hull
                cv2.drawContours(display_image, [full_cnt], -1, (0, 200, 255), 1)
                cv2.drawContours(display_image, [top_cnt], -1, (0, 255, 0), 2)
                cv2.circle(display_image, (top_cx, top_cy), 5, (0, 0, 255), -1)

                if cached_marker_xyz is not None:
                    robot_xyz, marker_xyz = camera_to_robot_frame(
                        peg_cam_xyz, cached_marker_xyz, cached_marker_yaw
                    )
                    rx, ry, rz = robot_xyz
                    coord_str = f"Robot XYZ: ({rx:.3f}, {ry:.3f}, {rz:.3f})m"

                    current_frame_pegs[color_label] = {
                        "shape": shape_name,
                        "pixel": (top_cx, top_cy),
                        "robot_xyz": robot_xyz,
                        "marker_xyz": marker_xyz,
                        "cam_xyz": peg_cam_xyz,
                    }
                else:
                    cx_m, cy_m, cz_m = peg_cam_xyz
                    coord_str = f"Cam XYZ: ({cx_m:.3f}, {cy_m:.3f}, {cz_m:.3f})m [No Marker]"

                cv2.putText(
                    display_image,
                    f"{color_label} ({shape_name})",
                    (top_cx - 50, top_cy - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display_image,
                    coord_str,
                    (top_cx - 60, top_cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                )

            # -----------------------------------------------------------------
            # STEP 3: Write to README.md Once Only
            # -----------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            ready_to_auto_save = (
                not readme_written
                and frame_count >= WARMUP_FRAMES
                and cached_marker_xyz is not None
                and len(current_frame_pegs) == EXPECTED_PEG_COUNT
            )
            manual_save_triggered = (
                not readme_written
                and key == ord("s")
                and cached_marker_xyz is not None
                and len(current_frame_pegs) > 0
            )

            if ready_to_auto_save or manual_save_triggered:
                base_cam_xyz = get_robot_base_in_camera_frame(
                    cached_marker_xyz, cached_marker_yaw
                )
                write_readme_once(
                    README_FILENAME,
                    cached_marker_id,
                    cached_marker_xyz,
                    base_cam_xyz,
                    current_frame_pegs,
                )
                readme_written = True
                if AUTO_EXIT_ON_SAVE:
                    break

            status_text = (
                "STATUS: SAVED TO README.md (Locked)"
                if readme_written
                else f"STATUS: Waiting for Marker + {EXPECTED_PEG_COUNT} Pegs ({len(current_frame_pegs)}/{EXPECTED_PEG_COUNT}) [Press 's' to save]"
            )
            status_color = (0, 255, 0) if readme_written else (0, 165, 255)
            cv2.putText(
                display_image,
                status_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
            )

            cv2.imshow("Peg Detection (Robot Base Frame)", display_image)
            if key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()