# W2V本体

from gensim.models.word2vec import Word2Vec
import pandas as pd

train = pd.read_csv('./traintestData/trainPreprocessed.csv')
test = pd.read_csv('./traintestData/testPreprocessed.csv')

vocab = list()
for j in range(2):
    if(j == 0):
        data = train
    elif(j == 1):
        data = test
    for i in range(data['id'].size):
        tempstr = data['CommentTxtWakati'][i].replace('[', '')
        tempstr = tempstr.replace(']', '')
        tempstr = tempstr.replace('\'', '')
        tempstr = tempstr.replace(',', '')
        tempstr = tempstr.split()
        vocab.append(tempstr)

# learn
print('start learning')
size = 300
model = Word2Vec(sentences=vocab, size=size, window=5, min_count=5, workers=4, iter=55)
modelname = './model/w2v' + str(size) + '.model'
model.save(modelname)
print('finished')
