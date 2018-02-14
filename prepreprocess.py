import pandas as pd
import pickle
import numpy as np
import tqdm

train = pd.read_csv('./traintestData/trainPreprocessed.csv')
test = pd.read_csv('./traintestData/testPreprocessed.csv')

data = train
vocabTr = list()
for i in range(data['id'].size):
    tempstr = data['CommentTxtWakati'][i].replace('[', '')
    tempstr = tempstr.replace(']', '')
    tempstr = tempstr.replace('\'', '')
    tempstr = tempstr.replace(',', '')
    tempstr = tempstr.split()
    vocabTr.append(tempstr)
data = test
vocabTe = list()
for i in range(data['id'].size):
    tempstr = data['CommentTxtWakati'][i].replace('[', '')
    tempstr = tempstr.replace(']', '')
    tempstr = tempstr.replace('\'', '')
    tempstr = tempstr.replace(',', '')
    tempstr = tempstr.split()
    vocabTe.append(tempstr)

with open('./traintestData/trainVocabPre.pickle', mode='wb') as f:
    pickle.dump(vocabTr, f)
with open('./traintestData/testVocabPre.pickle', mode='wb') as f:
    pickle.dump(vocabTe, f)

