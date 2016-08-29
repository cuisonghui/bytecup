#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from collections import defaultdict
from config.filepath_original import *

train = pd.read_csv(TRAIN, delimiter='\t', header=None)
train.columns = ('qid', 'uid', 'label')
train = train[~train.duplicated()]

user = pd.read_csv(USER, delimiter='\t', header=None)
user.columns = ('uid', 'ulabel', 'wordseq', 'charseq')
user = user[['uid', 'ulabel']]

df = pd.merge(user, train, on='uid')

dic_1 = defaultdict(int)
dic_total = defaultdict(int)
dic_rate = {}
for i in range(len(df)):
    line = df.iloc[i]
    ulabel = line.ulabel.split('/')
    ulabel = [int(x) for x in ulabel]
    for x in ulabel:
        dic_total[x] += 1
        if line.label == 1:
            dic_1[x] += 1

for k in dic_total.keys():
    dic_rate[k] = dic_1[k] / dic_total[k]
# ans_rate = pd.DataFrame(list(dic_rate.items()), columns=('ulabel', 'ansrate'))

valid = pd.read_csv(VALID)
pred = pd.merge(valid, user, on='uid', how='left')

for i in range(len(pred)):
    line = pred.iloc[i]
    if type(line.ulabel) != str:
        pred.loc[i, 'label'] = sum(dic_rate.values()) / len(dic_rate)
    else:
        ulabel = line.ulabel.split('/')
        ulabel = [int(x) for x in ulabel]
        pred.loc[i, 'label'] = sum([dic_rate[x] for x in ulabel]) / len(ulabel)

pred = pred[['qid', 'uid', 'label']]
pred.to_csv('output/baseline_user.csv', index=False)
