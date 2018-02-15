import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import ensemble
import pickle
import numpy as np
from downSampling import downSampling
import tqdm
from chainerHelper import ChainerHelper

def trans(vocab, model):
    # For文で最初に処理するデータがinvalidな場合if-elseで止まる不具合あり
    trainList = list()
    for l in tqdm.tqdm(vocab):
        fv = np.array([model.wv[d] for d in l if d in model.wv.vocab])
        if(fv.shape[0] == 0):
            fv = np.zeros_like(trainList[0])
        else:
            size = float(fv.shape[0])
            fv = np.sum(fv, axis=0) / size
        trainList.append(fv)
    return np.array(trainList)


if __name__ == '__main__':
    # prepare
    print('start transform traindata')
    trainList, label = downSampling(16225)

    # split
    ts = 0.3
    Xtra, Xte, Ytra, Yte = train_test_split(trainList, label, test_size=ts)

    # learn
    print('start learning')
    attrs = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    clfs = list()
    for idx, a in enumerate(attrs):
        clf = ensemble.RandomForestRegressor()
        clf.fit(Xtra, Ytra[:, idx])
        print('score: %.5f' % (clf.score(Xte, Yte[:, idx])))
        clfs.append(clf)

    # save model
    modelname = './model/RFReg.model'
    with open(modelname, 'wb') as f:
        pickle.dump(clfs, f)

    print('finished')
