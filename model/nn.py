#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Lambda, Merge, BatchNormalization, Input
from keras.callbacks import EarlyStopping
from keras import backend as K

from config import *
from config.filepath_original import *

def get_question_user():
    question = pd.read_csv(QUESTION, delimiter='\t', header=None)
    question.columns = ('qid', 'qlabel', 'qwordseq', 'qcharseq', 'likenum', 'ansnum', 'bansnum')
    qlabel = np.zeros((len(question), QLABEL_NUM))

    user = pd.read_csv(USER, delimiter='\t', header=None)
    user.columns = ('uid', 'ulabel', 'uwordseq', 'ucharseq')
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
    
    question = pd.concat([question.drop(['qlabel', 'qcharseq'], axis=1), qlabel], axis=1)
    user = pd.concat([user[['uid', 'uwordseq']], ulabel], axis=1)

    return question, user

def load_dateset(question, user):
    train = pd.read_csv(TRAIN, header=None, delimiter='\t')
    train.columns = ('qid', 'uid', 'label')
    train_x = train[['qid', 'uid']]
    train_y = train['label']
    train_x = pd.merge(train_x, question, on='qid', how='left')
    train_x = pd.merge(train_x, user, on='uid', how='left')
    train_x = train_x.drop(['qid', 'uid'], axis=1)

    test = pd.read_csv(VALID)
    test_x = test[['qid', 'uid']]
    test_x = pd.merge(test_x, question, on='qid', how='left') 
    test_x = pd.merge(test_x, user, on='uid', how='left')
    test_x = test_x.drop(['qid', 'uid'], axis=1)

    return (train_x, train_y), (test_x,)

def train_model(train, test):
    def transform(words):
        dic = {}
        result = []
        for word in words:
            a = []
            if word != '/':
                for idx in word.split('/'):
                    if idx not in dic:
                        dic[idx] = len(dic) + 1
                    a.append(dic[idx])
            result.append(a)
        return result

    train_x, train_y = train
    test_x = test[0]
    train_qwordseq = sequence.pad_sequences(transform(train_x['qwordseq']), maxlen=QWORDSEQ_MAXLEN)
    train_uwordseq = sequence.pad_sequences(transform(train_x['uwordseq']), maxlen=UWORDSEQ_MAXLEN)
    test_qwordseq = sequence.pad_sequences(transform(test_x['qwordseq']), maxlen=QWORDSEQ_MAXLEN)
    test_uwordseq = sequence.pad_sequences(transform(test_x['uwordseq']), maxlen=UWORDSEQ_MAXLEN)
    train_x = train_x.drop(['qwordseq', 'uwordseq'], axis=1).values
    test_x = test_x.drop(['qwordseq', 'uwordseq'], axis=1).values
    train_y = train_y.values
    print('train_x:', train_x.shape)
    print('train_y:', train_y.shape)
    print('test_x:', test_x.shape)

    def lambda_output_shape(input_shape):
        return (None, sum(shape[1] for shape in input_shape))

    batch_size = 128

    input_feature = Input(shape=(train_x.shape[1],))
    input_qwordseq = Input(shape=(QWORDSEQ_MAXLEN,))
    input_uwordseq = Input(shape=(UWORDSEQ_MAXLEN,))
    embedding_q = Embedding(QWORDSEQ_SIZE, 64, input_length=QWORDSEQ_MAXLEN, mask_zero=True)(input_qwordseq)
    embedding_u = Embedding(UWORDSEQ_SIZE, 64, input_length=UWORDSEQ_MAXLEN, mask_zero=True)(input_uwordseq)
    lstm_q = LSTM(64)(embedding_q)
    lstm_u = LSTM(64)(embedding_u) 
#    x = Merge([input_feature, lstm_q, lstm_u], mode='concat', concat_axis=1)()
    lambda_layer = Lambda(lambda x: K.concatenate(x, axis=1), output_shape=lambda_output_shape)
    x = lambda_layer([input_feature, lstm_q, lstm_u])
    x = BatchNormalization()(x)
    output = Dense(128, activation='sigmoid')(x)
    model = Model(input=[input_feature, input_qwordseq, input_uwordseq], output=output)

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit([train_x, train_qwordseq, train_uwordseq], train_y,
              batch_size=batch_size, nb_epoch=100, verbose=1,
              callbacks=[],
              validation_split=0.2,
              shuffle=True)

def main():
    question, user = get_question_user()
    train, test = load_dateset(question, user)
    train_model(train, test)

if __name__ == '__main__':
    main()
