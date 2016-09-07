#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split

from config import *
from config.filepath_original import *

def main():
    folder = sys.argv[1] if len(sys.argv) == 2 else 'datasets'

    question = pd.read_csv(os.path.join(folder, 'question_info.txt'), delimiter='\t', header=None)
    question.columns = ('qid', 'qlabel', 'wordseq', 'charseq', 'likenum', 'ansnum', 'bansnum')
    qlabel = np.zeros((len(question), QLABEL_NUM))

    user = pd.read_csv(os.path.join(folder, 'user_info.txt'), delimiter='\t', header=None)
    user.columns = ('uid', 'ulabel', 'wordseq', 'charseq')
    ulabel = np.zeros((len(user), ULABEL_NUM))

    for i, label in enumerate(question.qlabel):
        if label == '/':
            continue
        qlabel[i, int(label)] = 1
    qlabel = pd.DataFrame(qlabel, columns=('qlabel_{}'.format(i) for i in range(QLABEL_NUM)))

    for i, label in enumerate(user.ulabel):
        if label == '/':
            continue
        label = [int(x) for x in label.split('/')]
        ulabel[i, label] = 1
    ulabel = pd.DataFrame(ulabel, columns=('ulabel_{}'.format(i) for i in range(ULABEL_NUM)))
    
    question = pd.concat([question[['qid', 'likenum', 'ansnum', 'bansnum']], qlabel], axis=1)
    user = pd.concat([user[['uid']], ulabel], axis=1)

    train = pd.read_csv(TRAIN, delimiter='\t', header=None)
    train.columns = ('qid', 'uid', 'label')
    train_x = train[['qid', 'uid']]
    train_y = train['label']
    train_x = pd.merge(train_x, question, on='qid', how='left')
    train_x = pd.merge(train_x, user, on='uid', how='left')
    train_x = train_x.drop(['qid', 'uid'], axis=1)
    train_x = train_x.values
    train_y = train_y.values

    valid = pd.read_csv(VALID)
    valid = valid.drop('label', axis=1)
    valid = pd.merge(valid, question, on='qid', how='left')
    valid = pd.merge(valid, user, on='uid', how='left')
    valid = valid.drop(['qid', 'uid'], axis=1)

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=519)

    print(train_x.shape)
    print(train_y.shape)
    dtrain = xgb.DMatrix(train_x, train_y)
    dval = xgb.DMatrix(val_x, val_y)
    dtest = xgb.DMatrix(valid)
    params = {
        'booster': 'gbtree',
        'objective': 'rank:pairwise',
        'seed': 519,
        'scale_pos_weight': len(train_y[train_y==0]) / len(train_y[train_y==1]),
        'early_stopping_rounds': 100,
        'eval_metric': 'ndcg'
    }

    watchlist = [(dtrain, 'train'), (dval, 'val')]
    bst = xgb.train(params, dtrain, num_boost_round=2000, evals=watchlist)
    ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

    valid = pd.read_csv(VALID)
    valid = valid[['qid', 'uid']]
    pred = pd.concat([valid, pd.DataFrame(ypred, columns=['label'])], axis=1)
    pred.to_csv('output/xgb.csv', index=False)

if __name__ == '__main__':
    main()
