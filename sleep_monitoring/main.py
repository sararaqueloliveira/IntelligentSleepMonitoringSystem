import argparse
import os
import sys
import cv2
import imutils

from sleep_monitoring_v2.motion_detection import motion_detection
from sleep_status.sleep_status import face_recognition, sleep_status

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Vídeo de Input", default="../data/input/videos/video_30.mp4")
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
vs.set(cv2.CAP_PROP_FPS, 5)
print(vs.get(cv2.CAP_PROP_FPS))

# Initcialização das variáveis
i = 0
fgbg = cv2.createBackgroundSubtractorMOG2(300, 30, False)

# Processamento do video
for i in range(n_frames):
    frame = vs.read()
    frame = frame[1]
    frameId = vs.get(1)

    text = "Sem Movimento"

    # aplicação dos filtros
    blur = cv2.GaussianBlur(frame, (3, 3), 0)
    image = cv2.addWeighted(frame, 4, blur, -3, 0)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # get the foreground mask
    fg_mask = fgbg.apply(image)

    image = motion_detection.movement_detection(4000, image, fg_mask)
    cv2.imshow('Frame', image)
    cv2.imwrite('fotos/asleep' + str(i) + '.jpg', image)

    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break

vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
