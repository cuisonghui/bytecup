#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from config.filepath_original import *

train = pd.read_csv(TRAIN, delimiter='\t', header=None)
train.columns = ('qid', 'uid', 'label')

question = pd.read_csv(QUESTION, delimiter='\t', header=None)
question.columns = ('qid', 'qlabel', 'wordseq', 'charseq', 'likenum', 'ansnum', 'bansnum')
question = question[['qid', 'qlabel']]

df = pd.merge(question, train, on='qid')

group = df.groupby('qlabel')
ans_rate = []
for name, g in group:
    ans_rate.append([name, sum(g.label)/len(g)])
ans_rate = pd.DataFrame(np.array(ans_rate), columns=['qlabel', 'ansrate'])

df = pd.merge(question, ans_rate, on='qlabel')
df = df[['qid', 'ansrate']]
df = df[~df.duplicated()]

valid = pd.read_csv(VALID)
valid = valid.drop('label', axis=1)
pred = pd.merge(valid, df, on='qid')
pred.columns = ('qid', 'uid', 'label')
pred.to_csv('output/baseline_question.csv', index=False)
