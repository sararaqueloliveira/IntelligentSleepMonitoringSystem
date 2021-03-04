from __init__ import *
from collections import namedtuple
import numpy as np
import cv2

Detection = namedtuple("Detection", ["gt", "pred"])
video_path = "C:/Users/sarar/PycharmProjects/Tese_de_Mestrado_Imagem/sleepmonitoring/data/input/baby_awake_short.mp4"
file_path = "C:/Users/sarar/PycharmProjects/Tese_de_Mestrado_Imagem/sleepmonitoring/evaluation/Intersection_Over_Union/teste_1/prediction.txt"

if __name__ == "__main__":
	# definir um objeto para as deteções e para os resultados da iou
	detections = []
	iou_values = []

	vs = cv2.VideoCapture(video_path)

	# ler os valores da predicted box do ficheiro txt onde foram registados, frame por frame
	with open(file_path, "r") as a_file:
		for line in a_file:
			stripped_line = line.strip()
			# separar a string para array: "100, 100, 100, 100" => [100, 100, 100, 100]
			arrPred = [int(numeric_string) for numeric_string in stripped_line.split(',')]
			# adicionar ao array de deteções
			detections.append(Detection([237, 32, 367, 177], arrPred))

	for detection in detections:

		frame = vs.read()
		frame = frame if vs is None else frame[1]

		# desenhar a ground-truth box e a predicted box
		cv2.rectangle(frame, tuple(detection.gt[:2]), tuple(detection.gt[2:]), (0, 255, 0), 2)
		cv2.rectangle(frame, tuple(detection.pred[:2]), tuple(detection.pred[2:]), (0, 0, 255), 2)

		# calcular a intersection over union
		iou = bb_intersection_over_union(detection.gt, detection.pred)

		# mostrar o resultado em texto na frame e na consola
		cv2.putText(frame, "IoU: {:.4f}".format(iou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		print("{:.4f}".format(iou))
		iou_values.append(iou)
		
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(200) & 0xFF
		if key == ord("q"):
			break

	mean = calc_mean(iou_values)
	print("Média IOU: {:.4f}".format(mean))
	