#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
from sklearn.cross_validation import train_test_split

from config.filepath_original import *
from gen_group import gen_group

def main():
    df = pd.read_csv(TRAIN, header=None, delimiter='\t')
    df.columns = ('qid', 'uid', 'label')
    group = df.groupby('qid')
    group_qid = list(group.groups.keys())
    train_idx, val_idx = train_test_split(group_qid, test_size=0.2, random_state=519)

    for name, qids in zip(('train', 'val'), (train_idx, val_idx)):
        d = []
        for qid in qids:
            d.append(df.loc[group.groups[qid]])
        d = pd.concat(d)
        d.to_csv(os.path.join(DIR, name+'.csv'), index=False)
        gen_group(d, os.path.join(DIR, name+'.group'))

if __name__ == '__main__':
    main()
