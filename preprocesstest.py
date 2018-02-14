from gensim.models.word2vec import Word2Vec
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import tqdm

# 単語をW2Vで特徴量ベクトルに変換してPickle化
model = Word2Vec.load('./model/w2v.model')

# test data
print('start test transform')
with open('./traintestData/testVocabPre.pickle', mode='rb') as f:
    vocabTe = pickle.load(f)
testList = list()
for l in tqdm.tqdm(vocabTe):
    fv = np.array([model.wv[d]  for d in l if d in model.wv.vocab])
    testList.append(fv)
del(vocabTe)
del(model)
with open('./traintestData/testVocab.pickle', mode='wb') as f:
    pickle.dump(testList, f)

print('finished')
