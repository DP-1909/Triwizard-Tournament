import cv2
import numpy as np
import pyrealsense2 as rs


def nothing(x):
    pass


# Mouse callback to inspect exact HSV value on left-click
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_frame = param
        pixel = hsv_frame[y, x]
        print(
            f"Clicked Pixel ({x}, {y}) -> H: {pixel[0]}, S: {pixel[1]}, V: {pixel[2]}"
        )


def main():
    # Initialize RealSense pipeline (Color stream only for tuning)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # Create Trackbar Window
    cv2.namedWindow("HSV Tuner")
    cv2.createTrackbar("H Min", "HSV Tuner", 0, 179, nothing)
    cv2.createTrackbar("S Min", "HSV Tuner", 50, 255, nothing)
    cv2.createTrackbar("V Min", "HSV Tuner", 50, 255, nothing)
    cv2.createTrackbar("H Max", "HSV Tuner", 179, 179, nothing)
    cv2.createTrackbar("S Max", "HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("V Max", "HSV Tuner", 255, 255, nothing)

    cv2.namedWindow("Original Feed")
    print(
        "Controls:\n - Left-click on 'Original Feed' to sample pixel HSV\n - Press 'p' to print current bounds\n - Press 'q' to quit"
    )

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: 
                continue

            frame = np.asanyarray(color_frame.get_data())

            # Slight blur to reduce camera sensor noise
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # Bind updated HSV image to mouse click callback
            cv2.setMouseCallback("Original Feed", mouse_click, hsv)

            # Read trackbar positions
            h_min = cv2.getTrackbarPos("H Min", "HSV Tuner")
            s_min = cv2.getTrackbarPos("S Min", "HSV Tuner")
            v_min = cv2.getTrackbarPos("V Min", "HSV Tuner")
            h_max = cv2.getTrackbarPos("H Max", "HSV Tuner")
            s_max = cv2.getTrackbarPos("S Max", "HSV Tuner")
            v_max = cv2.getTrackbarPos("V Max", "HSV Tuner")

            lower_bound = np.array([h_min, s_min, v_min])
            upper_bound = np.array([h_max, s_max, v_max])

            # Create binary mask and clean up noise
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            result = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("Original Feed", frame)
            cv2.imshow("Mask", mask)
            cv2.imshow("HSV Tuner", result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):
                print(
                    f"\n'color_name': {{'lower': np.array([{h_min}, {s_min}, {v_min}]), 'upper': np.array([{h_max}, {s_max}, {v_max}])}},"
                )
            elif key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()