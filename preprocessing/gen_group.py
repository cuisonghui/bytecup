#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd

def gen_group(input_data, output_group):
    if isinstance(input_data, str):
        df = pd.read_csv(input_data, header=None, delimiter='\t')
        df.columns = ('qid', 'uid', 'label')
    elif isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        return

    qid = df.qid

    with open(output_group, 'w') as file:
        cur_qid = ''
        num = 0
        for q in qid:
            if cur_qid == q:
                num += 1
            else:
                if cur_qid != '':
                    file.write('{}\n'.format(num))
                cur_qid = q
                num = 1
        file.write('{}\n'.format(num))

def main():
    gen_group(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()

