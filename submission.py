import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pickle
import numpy as np
from chainerHelper import ChainerHelper


def getPred(y, clf):
    pred = clf.predict(y)
    return pred

if __name__ == '__main__':
    # testdata
    with open('./traintestData/testVocabW2VSUM.pickle', mode='rb') as f:
        testList = pickle.load(f)

    # load model
    # modelfile = './model/MultiElasticNet.model'
    # with open(modelfile, 'rb') as f:
    #    clf = pickle.load(f)
    clf = ChainerHelper(300, 6)
    modelfile = './model/ChainerNN.model'
    clf.load(modelfile)

    # predict
    predict = getPred(testList, clf)

    # transform into submission
    submissionData = pd.read_csv('./traintestData/testPreprocessed.csv')
    submissionData = submissionData.drop('CommentTxtWakati', axis=1)
    attrs = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for idx, val in enumerate(attrs):
        submissionData[val] = predict[:, idx]
    
    # save submission data
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    subfname = './submission/submission' + timestr + '.csv'
    submissionData.to_csv(subfname, index=False)
    
    print('finished')
