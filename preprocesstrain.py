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

with open('./traintestData/trainVocabW2VSUM.pickle', mode='wb') as f:
    pickle.dump(trainList, f)



print('finished')
