def downSampling(upperBound=16225):
    # ロジックがクソ過ぎて計算遅い
    import pandas as pd
    import random
    import pickle
    import numpy as np
    trainData = pd.read_csv('./traintestData/trainPreprocessed.csv')

    # 対象のIDを取得:validId
    extracted = trainData[trainData['tag'] == 'normal']
    extracted = extracted.reset_index(drop=True)
    sidx = random.sample(range(extracted.shape[0]), upperBound)
    sidx = sorted(sidx)
    validId = list()
    validId = [extracted['id'][i] for i in range(extracted.shape[0]) if(i in sidx)]

    # ToxicComment or validIdに入っているデータのみ使用する
    with open('./traintestData/trainVocabW2VSUM_1.pickle', mode='rb') as f:
        trainList = pickle.load(f)
    with open('./traintestData/trainVocabW2VSUM_2.pickle', mode='rb') as f:
        templist = pickle.load(f)
    trainList = np.vstack((trainList, templist))
    del(templist)

    with open('./traintestData/trainLabel.pickle', mode='rb') as f:
        label = pickle.load(f)
    validTrainList = list()
    validLabel = list()
    for i in range(trainData.shape[0]):
        if((trainData['tag'][i] != 'normal') or (trainData['id'][i] in validId)):
            validTrainList.append(trainList[i])
            validLabel.append(label[i])
    validTrainList = np.array(validTrainList)
    validLabel = np.array(validLabel)

    return validTrainList, validLabel
