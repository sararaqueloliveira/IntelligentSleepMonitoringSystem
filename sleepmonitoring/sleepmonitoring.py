import argparse
import os
import sys
import datetime
import cv2

from motion_detection.motion_detection import movement_detection

project_path = "C:/Users/sarar/PycharmProjects/Tese_de_Mestrado/sleepmonitoring"

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Vídeo de Input", default=project_path + "/data/input/videos/video_2.mp4")
ap.add_argument("-a", "--min-area", type=int, default=5000, help="Área mínima de movimento")
args = vars(ap.parse_args())

if ((not os.path.exists(args.get("video"))) or (not args.get("video"))):
    print("O ficheiro não existe!")
    sys.exit()

# Guardar a captura do vídeo na variável 'vs' e obter as suas dimensões
vs = cv2.VideoCapture(args["video"])
frame_width = int(vs.get(3))
frame_height = int(vs.get(4))
frameRate = vs.get(cv2.CAP_PROP_FPS)
n_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

# Incializar a gravação do vídeo output no computador
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(project_path + '/data/output/video_30.webm', fourcc, vs.get(cv2.CAP_PROP_FPS),
                      (frame_width, frame_height), 1)

# Criar um ficheiro para guardar as bounding-box estimadas
# f_prediction = open(project_path +  "/evaluation/testes/video_30/prediction.txt", "w+")
f = open(project_path + "/data/output/relatorio_movimentos.txt", "w+")

# Initcialização das variáveis
fgbg = cv2.createBackgroundSubtractorMOG2(300, 30, False)
sequence = 0
n_movements = 0
i = 0
prediction = Prediction()
movement = Movement()

# Processamento do video
while True:
    frame = vs.read()
    frame = frame[1]
    frameId = vs.get(1)

    text = "Sem Movimento"

    # terminado - relatório final
    if frame is None:
        write_final_report(n_frames, n_movements)
        break

    # aplicação dos filtros
    blur = cv2.GaussianBlur(frame, (3, 3), 0)

    image = cv2.addWeighted(frame, 4, blur, -3, 0)
    gray = cv2.cvtColor(cv2.UMat(image), cv2.COLOR_BGR2GRAY)

    # inicializar a primeira frame de background e passar para a próxima
    if not background:
        background = gray
        continue

    # verifica se existe movimento
    fgmask = fgbg.apply(image)

    image = movement_detection(2000, image, fgmask)
    move = movement.movement_detection(min_area=args["min_area"], gray=gray, background=background)

    if move:
        cv2.putText(frame, format("Movimento!"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        sequence = sequence + 1
        n_movements = n_movements + 1

        f.write(
            "Movimento | " + datetime.datetime.now().strftime("%A %d %B") + " | " + datetime.datetime.now().strftime(
                "%I:%M:%S%p") + "//n")

    # verifica o estado do sono
    (x1, y1, x2, y2), eyes = prediction.sleepstatus(image)
    text_eyes = "Sem deteção"

    if (x1, y1, x2, y2):
        if eyes == 1:
            text_eyes = "Olhos abertos"
        elif eyes == 2:
            text_eyes = "Olhos semi-fechados"
        else:
            text_eyes = "Olhos fechados"

    cv2.putText(frame, text_eyes, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

    # finalização
    text_status = 'Estado: a dormir'
    if move or eyes == 1:
        text_status = 'Estado: acordado'

    cv2.putText(frame, text_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    frame = cv2.resize(frame, (frame_width, frame_height))
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

f.close()
vs.stop() if args.get("video", None) is None else vs.release()
out.release()
cv2.destroyAllWindows()
