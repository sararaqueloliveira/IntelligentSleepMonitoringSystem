
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
	
