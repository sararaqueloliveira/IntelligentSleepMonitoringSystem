from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--video",
                help="path to input video", default="../input/man_movement_bed.mp4")
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="eyes_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Carregar os modelos
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt.txt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)
model = load_model(args["model"])

video = cv2.VideoCapture(args["video"])
frame_width = int(video.get(3))
frame_height = int(video.get(4))

# Incializar a gravação do vídeo output no computador
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter('../output/deeplearning/man_movement_bed.mp4', fourcc, video.get(cv2.CAP_PROP_FPS), (frame_width, frame_height), 1)

while True:
    frame = video.read()
    frame = frame[1]
    #frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(cv2.UMat(frame), cv2.COLOR_BGR2GRAY)

    (h, w) = frame.shape[:2]

    # Construir um blob da imagem e obter as deteções dos rostos encontrados
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Iterar sobre as deteções
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            # Obter as coordenadas para o retangulo onde está o rosto
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extrair o ROI, converter de bgr para rgb BGR e redimensionar
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)

            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # Obter uma predição para o frame de acordo com o modelo
            (Closed, Open) = model.predict(face)[0]

            # A partir dessa predição, escrever na frame se estão abertos ou fechados e a sua probabilidade
            label = "Open" if Open > Closed else "Closed"
            color = (0, 255, 0) if label == "Open" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(Open, Closed) * 100)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    frame = cv2.resize(frame, (frame_width, frame_height))
    cv2.imshow("Vídeo", frame)
    out.write(frame)

    key = cv2.waitKey(50) & 0xFF
    if key == ord("q"):
        break

video.stop() if args.get("video", None) is None else video.release()
out.release()
cv2.destroyAllWindows()
