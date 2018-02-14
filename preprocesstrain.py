from gensim.models.word2vec import Word2Vec
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import tqdm

# 単語をW2Vで特徴量ベクトルに変換してPickle化
model = Word2Vec.load('./model/w2v.model')


# train data
print('start transform traindata')
with open('./traintestData/trainVocabPre.pickle', mode='rb') as f:
    vocabTr = pickle.load(f)
trainList = list()
for l in tqdm.tqdm(vocabTr):
    fv = np.array([model.wv[d]  for d in l if d in model.wv.vocab])
    trainList.append(fv)
del(vocabTr)
del(model)
with open('./traintestData/trainVocab.pickle', mode='wb') as f:
    pickle.dump(trainList, f)

print('finished')
