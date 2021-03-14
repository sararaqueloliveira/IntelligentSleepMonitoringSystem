import numpy as np
import xml.etree.ElementTree as ET
import cv2
import math
import os


# função do cálculo do valor do IOU entre dois retângulos de regiôes
def bb_intersection_over_union(boxA, boxB):
	# determina as coordenadas (x, y) do retângulo de interseção
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# calculo da àrea de interseção
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# calculo das àreas do retângulo da predição e do ground-truth
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# calculo da IOU ao dividir a àrea do retângulo de interseção
	# e dividi-lo pela soma da predição + ground-truth - area de interseção
	# IOU = area of overlap / area of union
	iou = interArea / float(boxAArea + boxBArea - interArea)
	
	return iou
	
def calc_mean(array):
	return np.mean(array)

def xml_to_array(file_path):
	tree = ET.parse(file_path)
	root = tree.getroot()
	myArray=[]

	for x in root.findall('field'):
		myArray.append(x.text)

	print(myArray) 

def save_video_by_second(video, folder):
	path = "C://Users//sarar//PycharmProjects//Tese_de_Mestrado_Imagem//sleepmonitoring//data//output//evaluation//" + folder
	
	try:
		os.mkdir(path)
	except OSError:
		print ("Creation of the directory %s failed" % path)
	else:
		print ("Successfully created the directory %s " % path)

	cap = cv2.VideoCapture(video)
	frameRate = cap.get(cv2.CAP_PROP_FPS)  #frame rate
	print(frameRate);

	x = 1
	while(cap.isOpened()):
		frameId = cap.get(1) #current frame number
		print(frameId);
		ret, frame = cap.read()

		if (ret != True):
			break

		if (frameId % math.floor(frameRate) == 0):
			filename = path + "//" + str(int(x)) + ".jpg";
			x += 1
			cv2.imwrite(filename, frame)

	cap.release()
	print ("Done!")