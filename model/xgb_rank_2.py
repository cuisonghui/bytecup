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

def get_question_user():
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

    return question, user

def get_DataFrame(question, user, input_dataset, input_group, label=True):
    df = pd.read_csv(input_dataset)
    df_x = df[['qid', 'uid']]
    df_y = df['label']
    df_x = pd.merge(df_x, question, on='qid', how='left')
    df_x = pd.merge(df_x, user, on='uid', how='left')
    df_x = df_x.drop(['qid', 'uid'], axis=1)
    df_x = df_x.values
    df_y = df_y.values

    d = xgb.DMatrix(df_x, df_y) if label else xgb.DMatrix(df_x)
    d.set_group(np.loadtxt(input_group).astype(int))
    return d

def train_model(dtrain, dval, dtest):
    params = {
        'max_depth': 10,
        'eta': 0.02,
        'alpha': 0,
        'lambda': 400,
        'booster': 'gbtree',
        'objective': 'rank:pairwise',
        'seed': 519,
        'scale_pos_weight': 162326.0 / 21986.0,
        'early_stopping_rounds': 100,
        'eval_metric': 'ndcg'
    }

    watchlist = [(dtrain, 'train'), (dval, 'val')]
    bst = xgb.train(params, dtrain, num_boost_round=2000, evals=watchlist)
    return bst

def main():
    question, user = get_question_user()

    dtrain = get_DataFrame(question, user, os.path.join(DIR, 'train.csv'), os.path.join(DIR, 'train.group'))
    dval = get_DataFrame(question, user, os.path.join(DIR, 'val.csv'), os.path.join(DIR, 'val.group'))
    dtest = get_DataFrame(question, user, VALID, os.path.join(DIR, 'test.group'), label=False)

    bst = train_model(dtrain, dval, dtest)
    ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

    valid = pd.read_csv(VALID)
    valid = valid[['qid', 'uid']]
    pred = pd.concat([valid, pd.DataFrame(ypred, columns=['label'])], axis=1)
    pred.to_csv('output/xgb_rank_2.csv', index=False)

if __name__ == '__main__':
    main()
