import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import ensemble
import pickle
import numpy as np
from downSampling import downSampling, downSamplingEx
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
    # trainList, label = downSampling(16225)
    # ts = 0.3
    # Xtra, Xte, Ytra, Yte = train_test_split(trainList, label, test_size=ts)

    # learn
    print('start learning')
    attrs = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    dnum = [15294, 1595, 8449, 478, 7877, 1405]
    # dnum = [15294, 1800, 8449, 600, 7877, 1600]
    ts = 0.05
    clfs = list()
    for idx, a in enumerate(attrs):
        # テストデータは項目毎にDownsamplingしてみる
        trainList, label = downSamplingEx(dnum[idx], a)
        Xtra, Xte, Ytra, Yte = train_test_split(trainList, label, test_size=ts)
        # clf = ensemble.RandomForestRegressor()
        clf = ensemble.GradientBoostingRegressor(n_estimators=1000, verbose=True)
        clf.fit(Xtra, Ytra[:, idx])
        print('score: %.5f' % (clf.score(Xte, Yte[:, idx])))
        clfs.append(clf)

    # save model
    modelname = './model/GBRT' + str(ts) + '.model'
    with open(modelname, 'wb') as f:
        pickle.dump(clfs, f)

    print('finished')
