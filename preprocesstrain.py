from gensim.models.word2vec import Word2Vec
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import tqdm
from Learn import trans


# 単語をW2Vで特徴量ベクトルに変換してPickle化
model = Word2Vec.load('./model/w2v300.model')

# train data
print('start transform traindata')
with open('./traintestData/trainVocabPre.pickle', mode='rb') as f:
    vocabTr = pickle.load(f)
trainList = trans(vocabTr, model)
del(vocabTr)
del(model)

size = len(trainList) // 2
trainList_1 = trainList[0: size]
trainList_2 = trainList[size: len(trainList)]
del(trainList)

with open('./traintestData/trainVocabW2VSUM_1.pickle', mode='wb') as f:
    pickle.dump(trainList_1, f)
del(trainList_1)
with open('./traintestData/trainVocabW2VSUM_2.pickle', mode='wb') as f:
    pickle.dump(trainList_2, f)

print('finished')
