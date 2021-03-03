from __init__ import *
from collections import namedtuple
import numpy as np
import cv2

Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

if __name__ == "__main__":
	# definir um objeto para as deteções
	detections = []

	# ler os valores da predicted box do ficheiro txt onde foram registados, frame por frame
	with open("C:/Users/sarar/PycharmProjects/Tese_de_Mestrado_Imagem/sleepmonitoring/evaluation/Intersection_Over_Union/teste_1/prediction.txt", "r") as a_file:
		for line in a_file:
			stripped_line = line.strip()
			# separar a string para array: "100, 100, 100, 100" => [100, 100, 100, 100]
			arrPred = [int(numeric_string) for numeric_string in stripped_line.split(',')]
			# adicionar ao array de deteções
			detections.append(Detection("C:/Users/sarar/PycharmProjects/Tese_de_Mestrado_Imagem/sleepmonitoring/evaluation/Intersection_Over_Union/teste_1/frames/frame1.jpg", [237, 32, 367, 177], arrPred))

	# loop sobre as deteções
	for detection in detections:
		# carregar a frame em questão
		image = cv2.imread(detection.image_path)

		# desenhar a ground-truth box e a predicted box
		cv2.rectangle(image, tuple(detection.gt[:2]), tuple(detection.gt[2:]), (0, 255, 0), 2)
		cv2.rectangle(image, tuple(detection.pred[:2]), tuple(detection.pred[2:]), (0, 0, 255), 2)

		# calcular a intersection over union
		iou = bb_intersection_over_union(detection.gt, detection.pred)

		# mostrar o resultado em texto na frame e na consola
		cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		print("{}: {:.4f}".format(detection.image_path, iou))

		cv2.imshow("Image", image)
		cv2.waitKey(0)
