import cv2
import numpy as np
import pyrealsense2 as rs

# =====================================================================
# 1. PLACEHOLDER HSV THRESHOLDS (Replace with output from Script 1)
# =====================================================================
HSV_RANGES = {
    "Red_Peg": {
        "lower": np.array([0, 120, 70]),
        "upper": np.array([10, 255, 255]),
    },
    "Yellow_Peg": {
        "lower": np.array([20, 100, 100]),
        "upper": np.array([35, 255, 255]),
    },
    "Blue_Peg": {
        "lower": np.array([100, 150, 50]),
        "upper": np.array([130, 255, 255]),
    },
    "Purple_Peg": {
        "lower": np.array([135, 80, 50]),
        "upper": np.array([165, 255, 255]),
    },
}

# Minimum contour area (in pixels) to filter out noise
MIN_CONTOUR_AREA = 400


def classify_shape(contour):
    """Classifies contour into Triangle, Square, Hexagon, or Circle

    matching the 4 cutouts on the sorting board.
    """
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return "Unknown"

    # Approximate polygon vertices (tune 0.035 between 0.02 and 0.04 if needed)
    approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
    vertices = len(approx)

    # Circularity = 4 * pi * Area / Perimeter^2 (1.0 for a perfect circle)
    area = cv2.contourArea(contour)
    circularity = (4 * np.pi * area) / (peri * peri)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        return "Square"
    elif 5 <= vertices <= 6 and circularity < 0.86:
        return "Hexagon"
    elif vertices > 6 or circularity >= 0.86:
        return "Circle"
    return "Unknown"


def get_robust_depth(depth_image, depth_scale, cx, cy, kernel_size=5):
    """Samples a small patch around the centroid (cx, cy) and takes the median

    of non-zero depth pixels to avoid single-pixel depth dropouts.
    """
    h, w = depth_image.shape
    half_k = kernel_size // 2
    x_min, x_max = max(0, cx - half_k), min(w, cx + half_k + 1)
    y_min, y_max = max(0, cy - half_k), min(h, cy + half_k + 1)

    patch = depth_image[y_min:y_max, x_min:x_max]
    valid_pixels = patch[patch > 0]

    if len(valid_pixels) == 0:
        return 0.0

    # Return depth in meters
    return float(np.median(valid_pixels) * depth_scale)


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # Get depth scale (converts raw uint16 depth units to meters)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # CRITICAL: Align depth frame to color frame
    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            # Extract camera intrinsics for 2D -> 3D deprojection
            intrinsics = (
                color_frame.profile.as_video_stream_profile().intrinsics
            )

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            display_image = color_image.copy()

            blurred = cv2.GaussianBlur(color_image, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            kernel = np.ones((5, 5), np.uint8)

            # Iterate over each configured peg color
            for color_label, bounds in HSV_RANGES.items():
                mask = cv2.inRange(hsv, bounds["lower"], bounds["upper"])
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < MIN_CONTOUR_AREA:
                        continue

                    # Compute 2D pixel centroid (cx, cy)
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Classify geometric shape
                    shape_name = classify_shape(cnt)

                    # Get Z depth in meters and deproject to 3D Camera Frame (X, Y, Z)
                    depth_m = get_robust_depth(
                        depth_image, depth_scale, cx, cy
                    )
                    if depth_m == 0.0:
                        continue

                    # point_3d returns [X, Y, Z] in meters relative to camera optical center
                    point_3d = rs.rs2_deproject_pixel_to_point(
                        intrinsics, [cx, cy], depth_m
                    )
                    X_m, Y_m, Z_m = point_3d[0], point_3d[1], point_3d[2]

                    # Draw visual feedback
                    cv2.drawContours(display_image, [cnt], -1, (0, 255, 0), 2)
                    cv2.circle(display_image, (cx, cy), 4, (0, 0, 255), -1)

                    label = f"{color_label} ({shape_name})"
                    coord_text = (
                        f"X:{X_m:.3f}m Y:{Y_m:.3f}m Z:{Z_m:.3f}m"
                    )

                    cv2.putText(
                        display_image,
                        label,
                        (cx - 50, cy - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_image,
                        coord_text,
                        (cx - 50, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                    )

                    # Print or publish to ROS / controller
                    print(
                        f"[{color_label} | {shape_name}] -> Camera XYZ (m): ({X_m:.4f}, {Y_m:.4f}, {Z_m:.4f})"
                    )

            cv2.imshow("Peg Detection & 3D Localization", display_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()