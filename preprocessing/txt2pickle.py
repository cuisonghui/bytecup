#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle

dir_name = 'datasets'

user_dic = {}
with open(os.path.join(dir_name, 'user_info.txt')) as file:
    for line in file:
        line = line.split('\t')
        user_dic[line[0]] = {}
        
