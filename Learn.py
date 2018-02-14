import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pickle
import numpy as np
from downSampling import downSampling


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
    clf = linear_model.MultiTaskElasticNet(alpha=0.1)
    clf.fit(Xtra, Ytra)
    
    # score
    print('score: %.5f' % (clf.score(Xte, Yte)))
    
    # save model
    with open('./model/MultiElasticNet.model', 'wb') as f:
        pickle.dump(clf, f)
    print('finished')