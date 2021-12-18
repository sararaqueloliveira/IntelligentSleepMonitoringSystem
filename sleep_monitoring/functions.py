import numpy as np
import xml.etree.ElementTree as ET
import cv2
import math
import os
import re

project_path = "C://Users//sarar//PycharmProjects//Tese_de_Mestrado_Imagem//"


# Função do cálculo do valor do IOU entre dois retângulos de regiôes
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


# Cálculo da média de elementos de um array
def calc_mean(array):
    return np.mean(array)


# Guardar frames a cada segundo de vídeo
def save_video_by_second(video_path, folder_path):
    cap = cv2.VideoCapture(video_path)
    print(video_path)
    frameRate = cap.get(cv2.CAP_PROP_FPS)  # frame rate
    print(frameRate)

    x = 1
    while (cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        print(frameId)
        ret, frame = cap.read()

        if (ret != True):
            break

        if (frameId % math.floor(frameRate) == 0):
            filename = folder_path + "//" + str(int(x)) + ".jpg";
            x += 1
            cv2.imwrite(filename, frame)

    cap.release()
    print("Done!")


# Parse a um ficheiro xml, de modo a guardar os valores das anotações num ficheiro
def parse_xml_anotation_files(folder_rel_path):
    folder_path = project_path + folder_rel_path
    print(folder_path)
    file_pred = open(folder_path + "//" + "gt.txt", "w+")
    print(file_pred)

    files = get_ordered_files(folder_path)

    # Ordenação
    for filename in files:
        # Obter as coordenadas do ficheiro xml
        tree = ET.parse(folder_path + "//" + filename)
        root = tree.getroot()
        print(filename)

        coordinates = root.find('object').find('bndbox')

        xmin = coordinates.find('xmin').text
        ymin = coordinates.find('ymin').text
        xmax = coordinates.find('xmax').text
        ymax = coordinates.find('ymax').text

        file_pred.write(xmin + ', ' + ymin + ', ' + xmax + ', ' + ymax + '\n')


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def get_ordered_files(folder_path):
    files = []

    # Obter nome dos ficheiros
    for filename in os.listdir(folder_path):
        if filename.endswith(".xml"):
            files.append(filename)
        else:
            continue

    return sorted(files, key=numericalSort)


def prepare_evaluation(flag_anotacoes, flag_video):
    if (flag_anotacoes):
        i = 1
        while i <= 21:
            parse_xml_anotation_files("sleepmonitoring//evaluation//testes//video_" + str(i) + "//anotacoes//rosto")
            i = i + 1

    if (flag_video):
        filenames = []
        for path, dirs, files in os.walk(project_path + "sleepmonitoring//data//input"):
            filenames.append(files)

        i = 1
        if filenames:
            for video in filenames[0]:
                if video.endswith(".mp4") or video.endswith(".webm"):
                    folder_path = project_path + "sleepmonitoring//evaluation//testes//sleep_monitoring_v2" + str(i)
                    save_video_by_second(project_path + "sleepmonitoring//data//input//" + video,
                                         folder_path + "//frames")
                    i = i + 1


# Conclude the final report with the ratio of number of movements detected
def write_final_report(report, n_frames, n_movements):
    report.write("//nNumero Total de Movimentos: " + str(n_movements))
    report.write("//nNumero Total de Frames: " + str(n_frames))
    taxa = (n_movements / n_frames) * 100
    report.write("//nTaxa de Movimentacao durante o sono: " + str(taxa) + "%")