import argparse
import datetime
import imutils
import cv2
from prediction import Prediction
import math

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Vídeo de Input", default="C:/Users/sarar/PycharmProjects/Tese_de_Mestrado_Imagem/sleepmonitoring/data/input/child-going-to-sleep.mp4")
ap.add_argument("-a", "--min-area", type=int, default=5000, help="Área mínima de movimento")
args = vars(ap.parse_args())

# Guardar a captura do vídeo na variável 'vs' e obter as suas dimensões
vs = cv2.VideoCapture(args["video"])
frame_width = int(vs.get(3))
frame_height = int(vs.get(4))
frameRate = vs.get(cv2.CAP_PROP_FPS)

# Incializar a gravação do vídeo output no computador
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter('C:/Users/sarar/PycharmProjects/Tese_de_Mestrado_Imagem/sleepmonitoring/data/output/teste1.mp4', fourcc, vs.get(cv2.CAP_PROP_FPS), (frame_width, frame_height), 1)

# Criar um ficheiro de texto com o relatório dos movimentos do sono
f = open("C:/Users/sarar/PycharmProjects/Tese_de_Mestrado_Imagem/sleepmonitoring/data/output/relatorio_movimentos.txt", "w+")

# Criar um ficheiro para guardar as bounding-box estimadas
f_prediction = open("C://Users//sarar//PycharmProjects//Tese_de_Mestrado_Imagem//sleepmonitoring//evaluation//testes//teste4//prediction.txt", "w+")

# Inicializar as variáveis auxiliares no processamento do vídeo
background = None
sequence = 0

n_movimentos = 0
n_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

# Processar o vídeo frame por frame
i = 0
while True:
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]
    frameId = vs.get(1)

    text = "Sem Movimento"

    # Se não existir frame atual, quer dizer que o vídeo chegou ao fim e termina o ciclo
    if frame is None:
        f.write("//nNumero Total de Movimentos: " + str(n_movimentos))
        f.write("//nNumero Total de Frames: " + str(n_frames))
        taxa = (n_movimentos / n_frames) * 100
        f.write("//nTaxa de Movimentacao durante o sono: " + str(taxa) + "%")
        break

    # Redimensionar a frame de modo a torná-la mais pequena e tornar o processamento mais rápido
    # frame = imutils.resize(frame, width=500)
    # Converter para escalas de cinzento

    gray = cv2.cvtColor(cv2.UMat(frame), cv2.COLOR_BGR2GRAY)
    # Aplicar um filtro gaussiano (filtro de blur)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Incializar a frame de background inicial (primeira frame do vídeo)
    # Passar para a próxima frame pois não é preciso processamento na primeira
    if background is None:
        background = gray
        continue

    # Calculo da diferença absoluta entre a frame de background e a frame atual
    frameDelta = cv2.absdiff(background, gray)
    # Aplicação de um threshold binário à frame de diferença (irá ficar b&w)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilatar o threshold para preencher todas as falhas em volta
    thresh = cv2.dilate(thresh, None, iterations=10)
    # Obter os contornos do thresold
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Percorrer os contornos obtidos
    for c in cnts:
        # Se a área dos contornos é menor que a área definida, o movimento é insignificante
        if cv2.contourArea(c) < args["min_area"]:
            continue
        # Obter um retângulo representante do contorno/movimento e inseri-lo na frame atual
        (x, y, w, h) = cv2.boundingRect(c)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Movimento"
        # A variável sequecia incrementa a cada movimento e muda após x deteções
        # Atualizar background como a frame de movimento atual
        # A cada movimento, o background muda
        sequence = sequence + 1
        if sequence == 15:
            background = gray
            sequence = 0

    # Escrever o texto na frame (movimento / sem movimento)
    cv2.putText(frame, format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    if text == "Movimento":
        n_movimentos = n_movimentos + 1
        f.write("Movimento | " + datetime.datetime.now().strftime("%A %d %B") + " | " +
                datetime.datetime.now().strftime("%I:%M:%S%p") + "//n")

    # Localizar os olhos e detetar o seu estado (abertos/fechados/semi-abertos)
    prediction = Prediction()
    frame, face_box = prediction.predict_face(frame=frame)
    print(face_box)

    # Escrever num ficheiro as coordenadas da face

    if (frameId % math.floor(frameRate) == 0):
        f_prediction.write(face_box + "\n")
        cv2.imwrite('C://Users//sarar//PycharmProjects//Tese_de_Mestrado_Imagem//sleepmonitoring//evaluation//testes//teste5//frames//frame' + str(i) + '.jpg', frame)
        out.write(frame)
        i += 1

    frame = cv2.resize(frame, (frame_width, frame_height))
    # Mostrar o vídeo/frames
    cv2.imshow("Frame", frame)
    # Guardar o vídeo, frame por frame, no computador

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Fechar todas as janelas
f.close()
f_prediction.close()
vs.stop() if args.get("video", None) is None else vs.release()
out.release()
cv2.destroyAllWindows()
