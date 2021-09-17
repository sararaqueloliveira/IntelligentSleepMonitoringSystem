from __init__ import *
import pandas as pd
from sklearn.metrics import recall_score, precision_score, confusion_matrix, accuracy_score, f1_score

df = pd.read_excel('C://Users//sarar//PycharmProjects//Tese_de_Mestrado_Imagem//sleepmonitoring//evaluation//dados//dados_v2.0.xlsx', "video_31_acordar")
df.head()

accuracy = accuracy_score(df.ground_truth.values, df.predicted.values)
recall = recall_score(df.ground_truth.values, df.predicted.values)
precision = precision_score(df.ground_truth.values, df.predicted.values, average='weighted', labels=np.unique(df.predicted.values))
f1_score = f1_score(df.ground_truth.values, df.predicted.values)

#confusion_matrix = confusion_matrix(df.ground_truth.values, df.predicted.values)

print('Accuracy: %.3f'%(accuracy))
print('Precision: %.3f'%(precision))
print('Recall: %.3f'%(recall))
print('F1: %.3f'%(f1_score))

# print(confusion_matrix)


