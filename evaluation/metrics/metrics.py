import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, confusion_matrix, accuracy_score, f1_score

df = pd.read_excel('../../evaluation/audio_recognition_module/dados.xlsx', "1")
df.head()
print(df.ground_truth.values)
print(df.predicted.values)
accuracy = accuracy_score(df.ground_truth.values, df.predicted.values)
recall = recall_score(df.ground_truth.values, df.predicted.values, labels=np.unique(df.predicted.values), zero_division=1)
precision = precision_score(df.ground_truth.values, df.predicted.values, labels=np.unique(df.predicted.values), zero_division=1)
f1_score = f1_score(df.ground_truth.values, df.predicted.values, labels=np.unique(df.predicted.values), zero_division=1)

#confusion_matrix = confusion_matrix(df.ground_truth.values, df.predicted.values)

print('%.3f ' % accuracy + '%.3f ' % precision + '%.3f ' % recall + '%.3f ' % f1_score)
print('Precision: %.3f' % precision)
print('Recall: %.3f' % recall)
print('F1: %.3f' % f1_score)

# print(confusion_matrix)


