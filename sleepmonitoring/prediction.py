from keras.models import load_model
import numpy as np
import keras
from imutils import face_utils
import dlib
import cv2
import face_recognition


class Prediction:
    left_eye_start_index, left_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    right_eye_start_index, right_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    facial_landmarks_predictor = './classifiers/68_face_landmarks_predictor.dat'
    predictor = dlib.shape_predictor(facial_landmarks_predictor)

    model = load_model('classifiers/weights.149-0.01.hdf5')

    def predict_eye_state(self, image):
        image = cv2.resize(image, (20, 10))
        image = image.astype(dtype=np.float32)

        image_batch = np.reshape(image, (1, 10, 20, 1))
        image_batch = keras.applications.mobilenet.preprocess_input(image_batch)
        return np.argmax(self.model.predict(image_batch)[0])

    def predict_face(self, frame):
        face_box = "0, 0, 0, 0"
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        # resized_image = cv2.resize(image, (500, 500))
        lab = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
        l, _, _ = cv2.split(lab)
        resized_height, resized_width = l.shape[:2]
        height_ratio, width_ratio = original_height / resized_height, original_width / resized_width

        # Reconhecer os rostos presentes na frame
        face_locations = face_recognition.face_locations(l, model='hog')

        # Se for reconhecida algum rosto
        if len(face_locations):
            top, right, bottom, left = face_locations[0]
            x1, y1, x2, y2 = left, top, right, bottom

            x1 = int(x1 * width_ratio)
            y1 = int(y1 * height_ratio)
            x2 = int(x2 * width_ratio)
            y2 = int(y2 * height_ratio)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shape = self.predictor(gray, dlib.rectangle(x1, y1, x2, y2))

            face_landmarks = face_utils.shape_to_np(shape)

            left_eye_indices = face_landmarks[self.left_eye_start_index:
                                              self.left_eye_end_index]

            (x, y, w, h) = cv2.boundingRect(np.array([left_eye_indices]))

            left_eye = gray[y:y + h, x:x + w]

            right_eye_indices = face_landmarks[self.right_eye_start_index:
                                               self.right_eye_end_index]

            (x, y, w, h) = cv2.boundingRect(np.array([right_eye_indices]))
            right_eye = gray[y:y + h, x:x + w]

            left_eye_open = 'yes' if self.predict_eye_state(image=left_eye) else 'no'
            right_eye_open = 'yes' if self.predict_eye_state(image=right_eye) else 'no'

            if left_eye_open == 'yes' and right_eye_open == 'yes':
                cv2.putText(frame, "Olhos abertos - 100%", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif left_eye_open == 'yes' and right_eye_open == 'no' or left_eye_open == 'no' and right_eye_open == 'yes':
                cv2.putText(frame, "Olhos semi-fechados", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Olhos fechados", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            face_box = str(x1) + ', ' + str(y1) + ', ' + str(x2) + ', ' + str(y2)

        return frame, face_box
