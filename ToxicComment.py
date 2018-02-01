# coding: utf-8

import pandas as pd
from tqdm import tqdm
import treetaggerwrapper
import os
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def wakatiWithTT(txt):
    tagdir = os.getenv('TREETAGGER_ROOT')
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR=tagdir)
    tags = tagger.TagText(txt)
    validPos = ['VV', 'NN', 'NP', 'JJ', 'VH']
    extractedTag = list()
    for tag in tags:
        sptag = tag.split()
        if (sptag[1][0: 2] in validPos):
            extractedTag.append(sptag[2])
    return extractedTag


if __name__ == '__main__':
    # train Data
    data = pd.read_csv('./data/train.csv')
    attrs = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    tagdir = os.getenv('TREETAGGER_ROOT')
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR=tagdir)
    wakati = [wakatiWithTT(d) for d in tqdm(data['comment_text'])]
    data['CommentTxtWakati'] = pd.Series(wakati)
    data.to_csv('./data/train_wakated.csv')
