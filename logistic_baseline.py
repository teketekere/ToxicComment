import numpy as np
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import pickle

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('./data/train.csv').fillna(' ')
test = pd.read_csv('./data/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])


wordpath = './traintestData/word_vectorizer.pickle'
if(os.path.isfile(wordpath)):
    with open(wordpath, 'rb') as f:
        word_vectorizer = pickle.load(f)
else:
    print("start word learn")
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 2),
        max_features=20000)
    word_vectorizer.fit(all_text)
    with open(wordpath, 'wb') as f:
        pickle.dump(word_vectorizer, f)
charpath = './traintestData/char_vectorizer.pickle'
if(os.path.isfile(charpath)):
    with open(charpath, 'rb') as f:
        char_vectorizer = pickle.load(f)
else:
    print("start char learn")
    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        stop_words='english',
        ngram_range=(2, 6),
        max_features=30000)
    char_vectorizer.fit(all_text)
    with open(charpath, 'wb') as f:
        pickle.dump(char_vectorizer, f)

trainTfidfPath = './traintestData/tfidfTrain.pickle'
testTfidfPath = './traintestData/tfidfTest.pickle'
if(os.path.isfile(trainTfidfPath) and os.path.isfile(testTfidfPath)):
    with open(trainTfidfPath, 'rb') as f:
        train_features = pickle.load(f)
    with open(testTfidfPath, 'rb') as f:
        test_features = pickle.load(f)
else:
    print("get word features")
    train_word_features = word_vectorizer.transform(train_text)
    test_word_features = word_vectorizer.transform(test_text)
    print("get char features")
    train_char_features = char_vectorizer.transform(train_text)
    test_char_features = char_vectorizer.transform(test_text)
    print("get features")
    train_features = hstack([train_char_features, train_word_features])
    test_features = hstack([test_char_features, test_word_features])
    with open(trainTfidfPath, 'wb') as f:
        pickle.dump(train_features, f)
    with open(testTfidfPath, 'wb') as f:
        pickle.dump(test_features, f)

clfname = "gbt"
print("start fitting")
scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    train_target = train[class_name]
    # classifier = LogisticRegression(solver='sag')
    classifier = ensemble.GradientBoostingRegressor(n_estimators=500, verbose=True)
    # cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    # scores.append(cv_score)
    # print('CV score for class {} is {}'.format(class_name, cv_score))
    classifier.fit(train_features, train_target)
    # submission[class_name] = classifier.predict_proba(test_features)[:, 1]
    submission[class_name] = classifier.predict(test_features)

# print('Total CV score is {}'.format(np.mean(scores)))

import time
timestr = time.strftime("%Y%m%d-%H%M%S")
submissionName = './submission/' + clfname + timestr + '.csv'
submission.to_csv(submissionName, index=False)
