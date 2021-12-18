import argparse
import os
import sys
import cv2
import datetime
import matplotlib.pyplot as plt
from audio_recognition.functions import *
from scipy.io import wavfile

from motion_detection.motion_detection import movement_detection
from sleep_status.sleep_status import sleep_status
from functions import write_final_report

video = 'videos/video_30'
audio = 'audio/audio_30'
ext = 'mp4'

ap = argparse.ArgumentParser()
ap.add_argument("--file_type", help="File Type [video | audio]", default="video")
ap.add_argument("--file_path", help="File Path", default="../data/input/" + video + "." + ext)
ap.add_argument("--min_movement_area",  help="Minimum movement area", default=2000)
ap.add_argument("--pose_estimation",  help="Activate pose estimation [0 | 1]", default=0)
args = vars(ap.parse_args())

file_type = args.get("file_type")
file_path = args.get("file_path")
min_movement_area = args.get("min_movement_area")
pose_estimation_flag = args.get("pose_estimation")

# Check if the file exists
if not os.path.exists(file_path):
    print("The file does not exist. Please enter a valid file.")
    sys.exit()

# Open file for final report
report = open("../data/output/final_report.txt", "w+")

# Check file type
if file_type == 'video':
    # Read video file
    vs = cv2.VideoCapture(file_path)
    frameRate = vs.get(cv2.CAP_PROP_FPS)
    n_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize auxiliary variables
    second = 0
    j = 0
    n_movements = 0

    # Creation the background subtractor for the Motion Detection Module
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(300, 30, False)

    # Video processing
    for i in range(n_frames):
        frame = vs.read()
        frame = frame[1]
        frameId = vs.get(1)

        # End of the video
        if frame is None:
            # Conclude the final report with the ratio of number of movements detected
            write_final_report(report, n_frames, n_movements)
            break

        # Discard the first 50 frames
        if frameId < 50.0:
            continue

        # Applying filters to the frame
        blur = cv2.GaussianBlur(frame, (3, 3), 0)
        sharpen = cv2.addWeighted(frame, 4, blur, -3, 180)
        gray = cv2.cvtColor(sharpen, cv2.COLOR_RGB2GRAY)

        # MOTION DETECTION MODULE #
        fg_mask = bg_subtractor.apply(gray)
        movement, body_parts, frame = movement_detection(min_movement_area, fg_mask, frame, pose_estimation_flag)
        # Movement occurred
        if movement:
            cv2.putText(frame, format("Movement!"), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 130), 2)
            n_movements = n_movements + 1

            # Write in final report
            report.write("Movement | " + datetime.datetime.now().strftime("%A %d %B") + " | " +
                         datetime.datetime.now().strftime("%I:%M:%S%p") + "\n\n")

        # SLEEP STATUS DETECTION MODULE #
        face_box, status = sleep_status(sharpen, frame, 0)
        if face_box != "0, 0, 0, 0":
            # Face found
            cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 0, 0), 2)
            # Eye status
            cv2.putText(frame, "Sleep status: " + status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 130), 2)
            # Write in the final report if the person woke up
            report.write("Awake | " + datetime.datetime.now().strftime("%A %d %B") + " | " +
                         datetime.datetime.now().strftime("%I:%M:%S%p") + "\n\n")

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

    vs.stop() if args.get("video", None) is None else vs.release()
    cv2.destroyAllWindows()

elif file_type == 'audio':
    # Load yamnet retrained model
    saved_model_path = '../models/yamnet_retrained'
    reloaded_model = tf.saved_model.load(saved_model_path)

    my_classes = ['snoring', 'breathing']

    wav_file_name = file_path
    sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

    # Basic information about the audio
    duration = len(wav_data) / sample_rate
    print("Audio informations:")
    print(f'Sample rate: {sample_rate} Hz')
    print(f'Total duration: {duration:.2f}s')
    print(f'Size of the input: {len(wav_data)}')

    # Obtain classes scores for the audio
    waveform = load_wav_16k_mono(wav_file_name)
    scores = reloaded_model(waveform)
    your_top_class = tf.argmax(scores)
    your_infered_class = my_classes[your_top_class]
    class_probabilities = tf.nn.softmax(scores, axis=-1)
    your_top_score = class_probabilities[your_top_class]

    print(f'The sound detected is: {your_infered_class} ({your_top_score})')

    # Write in the final report if the main class detected is <snoring>
    report.write("Snoring | Probability: " + str('{:.2f}'.format(your_top_score)) + " | " + datetime.datetime.now().strftime("%A %d %B") + " | " +
                 datetime.datetime.now().strftime("%I:%M:%S%p") + "//n")

    # Build probabilities bar graph
    probabilities = class_probabilities.numpy()
    plt.barh(my_classes, probabilities, color=['black'], height=0.2)
    plt.title('Audio classification score')
    plt.ylabel('Classes')
    plt.xlabel('Score')
    plt.show()

report.close()
