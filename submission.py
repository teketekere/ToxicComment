import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pickle
import numpy as np
from chainerHelper import ChainerHelper


def getPred(y, clfs):
    attrs = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    preds = list()
    for idx, a in enumerate(attrs):
        pred = clfs[idx].predict(y)
        preds.append(pred)
    return np.array(preds)

if __name__ == '__main__':
    # testdata
    with open('./traintestData/testVocabW2VSUM.pickle', mode='rb') as f:
        testList = pickle.load(f)

    # load model
    modelfile = './model/GBRT0.2.model'
    with open(modelfile, 'rb') as f:
        clfs = pickle.load(f)

    # predict
    predict = getPred(testList, clfs)
    # transform into submission
    submissionData = pd.read_csv('./traintestData/testPreprocessed.csv')
    submissionData = submissionData.drop('CommentTxtWakati', axis=1)
    attrs = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for idx, val in enumerate(attrs):
        submissionData[val] = predict[idx]

    # save submission data
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    subfname = './submission/submission' + timestr + '.csv'
    submissionData.to_csv(subfname, index=False)

    print('finished')
