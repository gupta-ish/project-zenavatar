import pyrealsense2 as rs
import numpy as np
import cv2
import os

print("Initializing...")

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale = ", depth_scale)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = "realsense_video.avi"
video_writer = cv2.VideoWriter(output_file, fourcc, 30, (640, 480))


os.makedirs("output_images", exist_ok=True)

print("Recording started. Press 'Ctrl+C' to stop.")

try:
    frame_count = 0
    while True:
        frames = pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        rgb_file = f"output_images/rgb_{frame_count:04d}.png"
        depth_file = f"output_images/depth_{frame_count:04d}.png"
        combined_file = f"output_images/combined_{frame_count:04d}.png"

        cv2.imwrite(rgb_file, color_image)
        cv2.imwrite(depth_file, depth_colormap)

        combined_image = np.hstack((color_image, depth_colormap))
        cv2.imwrite(combined_file, combined_image)

        print(f"Saved frame {frame_count}: RGB and Depth images.")
        video_writer.write(color_image)

        frame_count += 1

except KeyboardInterrupt:
    print("Recording stopped by user.")

finally:
    pipeline.stop()
    video_writer.release()
    print(f"Video saved as {output_file}.")
    print("Images saved in 'output_images' directory.")

