import argparse
import os
import sys
import cv2
import math

from motion_detection.motion_detection import movement_detection
from sleep_status.sleep_status import face_recognition, sleep_status

video = 'video_37'
ext = 'webm'

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Vídeo de Input", default="../data/input/videos/" + video + "." + ext)
ap.add_argument("-a", "--min-area", type=int, default=5000, help="Área mínima de movimento")
args = vars(ap.parse_args())

if (not os.path.exists(args.get("video"))) or (not args.get("video")):
    print("O ficheiro não existe!")
    sys.exit()

# Guardar a captura do vídeo na variável 'vs' e obter as suas dimensões
vs = cv2.VideoCapture(args["video"])
frame_width = int(vs.get(3))
frame_height = int(vs.get(4))
frameRate = vs.get(cv2.CAP_PROP_FPS)
n_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

# Initcialização das variáveis
i = 0
fgbg = cv2.createBackgroundSubtractorMOG2(300, 30, False)

# Files
f_iou = open("../evaluation/testes/" + video + "/prediction.txt", "w+")

# Processamento do video
second = 0
j = 0
for i in range(n_frames):
    frame = vs.read()
    frame = frame[1]
    frameId = vs.get(1)


    # aplicação dos filtros
    blur = cv2.GaussianBlur(frame, (3, 3), 0)

    # só para face detection
    #image = cv2.addWeighted(frame, 4, blur, -3, 0)
    image = cv2.addWeighted(frame, 4, blur, -3, 150)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # get the foreground mask
    fg_mask = fgbg.apply(image)

    #image = movement_detection(2000, image, fg_mask)
    #image = sleep_status(gray, image)

    if frameId % math.floor(frameRate) == 0:
        second = 1

    #image = movement_detection(2000, image, fg_mask, second)
   # face_box, frame = sleep_status(gray, frame)
   # cv2.imshow('Frame', frame)

    if second:
        j = j + 1
        face_box, frame = sleep_status(gray, frame)
        #cv2.imwrite('../evaluation/testes/' + video + '/predicoes/' + str(j) + '.jpg', image)
        f_iou.write(face_box + "\n")



    second = 0
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break

vs.stop() if args.get("video", None) is None else vs.release()
f_iou.close()
cv2.destroyAllWindows()
