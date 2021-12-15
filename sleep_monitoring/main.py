import argparse
import os
import sys
import cv2
import math

from motion_detection.motion_detection import movement_detection
from sleep_status.sleep_status import face_recognition, sleep_status

video = 'video_30'
ext = 'mp4'

ap = argparse.ArgumentParser()
ap.add_argument("--file_type", help="File Type [video | audio]", default="video")
ap.add_argument("--file_path", help="File Path", default="../data/input/videos/" + video + "." + ext)
ap.add_argument("--min_movement_area",  help="Minimum movement area", default=2000)
ap.add_argument("--pose_estimation",  help="Activate pose estimation [0 | 1]", default=1)
args = vars(ap.parse_args())

file_type = args.get("file_type")
file_path = args.get("file_path")
min_movement_area = args.get("min_movement_area")
print(min_movement_area)
pose_estimation_flag = args.get("pose_estimation")

# Check if the file exists
if not os.path.exists(file_path):
    print("The file does not exist. Please enter a valid file.")
    sys.exit()

# Check file type
if file_type == 'video':
    # Read video file
    vs = cv2.VideoCapture(file_path)
    frameRate = vs.get(cv2.CAP_PROP_FPS)
    n_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize auxiliary variables
    second = 0
    j = 0

    # Creation the background subtractor for the Motion Detection Module
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(300, 30, False)

    # Video processing
    for i in range(n_frames):
        frame = vs.read()
        frame = frame[1]
        frameId = vs.get(1)

        # Discard the first 50 frames
        if frameId < 50.0:
            continue

        # Applying filters to the frame
        blur = cv2.GaussianBlur(frame, (3, 3), 0)
        #sharpen = cv2.addWeighted(frame, 4, blur, -3, 0)
        sharpen = cv2.addWeighted(frame, 4, blur, -3, 180)
        gray = cv2.cvtColor(sharpen, cv2.COLOR_RGB2GRAY)

        # Sleep status detection module
        face_box, image_1 = sleep_status(sharpen, frame)

        # Motion detection module
        fg_mask = bg_subtractor.apply(gray)
        movement, body_parts, image_2 = movement_detection(min_movement_area, fg_mask, frame, pose_estimation_flag)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

    vs.stop() if args.get("video", None) is None else vs.release()
    cv2.destroyAllWindows()
