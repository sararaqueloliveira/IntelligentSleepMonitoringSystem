from keras.models import load_model
import numpy as np
import keras
from imutils import face_utils
import dlib
import cv2
from scipy.spatial import distance as dist

left_eye_start_index, left_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
right_eye_start_index, right_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
MINIMUM_EAR = 0.4

detector = dlib.get_frontal_face_detector()
facial_landmarks_predictor = '../models/face_predictor/68_face_landmarks_predictor.dat'
predictor = dlib.shape_predictor(facial_landmarks_predictor)

model = load_model('../models/face_predictor/weights.149-0.01.hdf5')


def sleep_status(gray, frame):
    frame = face_recognition(gray, frame)

    return frame


# Second step: Eye state recognition
def predict_eye_state(landmarks, image):
    # coordinates of the landmarks previously detected
    leftEye = landmarks[left_eye_start_index:left_eye_end_index]
    rightEye = landmarks[right_eye_start_index:right_eye_end_index]

    # EAR calculation for both eyes
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    # average the EAR together for both eyes
    ear = (leftEAR + rightEAR) / 2.0

    # check if the EAR calculated is lower or higher than the pre-defined minimum EAR
    # if EAR is higher than minimum EAR: state = awake
    # if EAR is lower than minimum EAR: state = sleep
    if ear < MINIMUM_EAR:
        state = 'sleeping'
    else:
        state = 'awake'

    cv2.putText(image, "Sleep state: " + state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 130), 2)
    return image


# First step: Face recognition in frame
def face_recognition(gray, frame):
    # the detector will find the bounding box of the face found in the frame
    dets = detector(gray, 1)

    # Through the detected bounding box, the landing marks of the face are detected
    for (i, dect) in enumerate(dets):
        shape = predictor(gray, dect)
        face_landmarks = face_utils.shape_to_np(shape)

        # draw the face bounding box through the coordinates
        (x, y, w, h) = face_utils.rect_to_bb(dect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

        # find the coordinates for the facial landmarks and draw them
        # for (x, y) in face_landmarks:
        #    cv2.circle(frame, (x, y), 1, (0, 0, 117), -2)

        frame = predict_eye_state(face_landmarks, frame)

    return frame


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the points:
    # |p2 - p6|
    dif_1 = dist.euclidean(eye[1], eye[5])
    # |p3 - p5|
    dif_2 = dist.euclidean(eye[2], eye[4])
    # |p1 - p4|
    dif_3 = dist.euclidean(eye[0], eye[3])

    # eye aspect ratio
    ear = (dif_1 + dif_2) / (2.0 * dif_3)
    return ear
