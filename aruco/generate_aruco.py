import cv2
import numpy as np
import os
# - Marker 1: (Top-left corner) ID = 0
# - Marker 2: (Top-right corner) ID = 1
# - Marker 3: (Bottom-left corner) ID = 2
# - Marker 4: (Bottom-right corner) ID = 3


# ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# # Define marker ID and size
# MARKER_ID = 0
# MARKER_SIZE = 300  # Pixels

# # Generate marker image
# marker_img = cv2.aruco.generateImageMarker(ARUCO_DICT, MARKER_ID, MARKER_SIZE)


# # Save the marker image
# cv2.imwrite("aruco_marker.png", marker_img)

# # Display the marker
# cv2.imshow("ArUco Marker", marker_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

MARKER_IDS = [0, 1, 2, 3]  
MARKER_SIZE = 300  
output_dir = "aruco_markers"
os.makedirs(output_dir, exist_ok=True)

for marker_id in MARKER_IDS:
    marker_img = cv2.aruco.generateImageMarker(ARUCO_DICT, marker_id, MARKER_SIZE)
    marker_filename = os.path.join(output_dir, f"aruco_marker_{marker_id}.png")
    cv2.imwrite(marker_filename, marker_img)

    cv2.imshow(f"ArUco Marker {marker_id}", marker_img)
    cv2.waitKey(500)  

cv2.destroyAllWindows()

print(f"Markers saved in '{output_dir}' directory.")