import pyrealsense2 as rs
import numpy as np
import cv2


print("init ...")


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = "realsense_video.avi"
video_writer = cv2.VideoWriter(output_file, fourcc, 30, (640, 480))

print("Recording started. Press 'q' to stop.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # cv2.imshow('RealSense Video', color_image)
        cv2.imwrite("output.png", color_image)

        video_writer.write(color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {output_file}")

