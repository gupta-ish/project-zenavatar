import pyrealsense2 as rs
import numpy as np
import cv2
import os

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_LENGTH = 0.1  # meters
OUTPUT_FILE = "pose_output.txt"

# Setup RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Load ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()

# Intrinsic calibration (will be fetched from RealSense)
profile = pipeline.get_active_profile()
color_stream = profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                          [0, intr.fy, intr.ppy],
                          [0, 0, 1]])
dist_coeffs = np.array(intr.coeffs)

print("Camera intrinsics:")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# Output file setup
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

# Main loop
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                # Draw marker and axis
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
                cv2.aruco.drawAxis(color_image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

                # Write pose to file
                with open(OUTPUT_FILE, "a") as f:
                    f.write(f"ID: {ids[i][0]}\n")
                    f.write(f"Translation (tvec): {tvecs[i].flatten().tolist()}\n")
                    f.write(f"Rotation (rvec): {rvecs[i].flatten().tolist()}\n\n")

        # Show result
        cv2.imshow("ArUco Detection", color_image)
        key = cv2.waitKey(1)
        if key == 27:  # Esc key
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"\nPose data saved to '{OUTPUT_FILE}'")
