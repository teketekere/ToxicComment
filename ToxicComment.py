# coding: utf-8

from multiprocessing import Pool
import multiprocessing as multi
import pandas as pd
import tqdm
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
    [extractedTag.append(tag.split()[2]) for tag in tags if tag.split()[1][0:2] in validPos]
    '''
    for tag in tags:
        sptag = tag.split()
        if (sptag[1][0:2] in validPos):
            extractedTag.append(sptag[2])
    '''
    return extractedTag


if __name__ == '__main__':
    # get train Data
    data = pd.read_csv('./data/train.csv', encoding='utf-8')
    attrs = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Preprocessing train Data
    with Pool(multi.cpu_count()) as p:
        wakati = list(tqdm.tqdm(p.imap(wakatiWithTT, data['comment_text']), total=data['comment_text'].size))
    data['CommentTxtWakati'] = pd.Series(wakati)
    data.to_csv('./data/train_wakated.csv')
