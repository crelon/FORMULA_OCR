#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-12 12:03:05
# Last Modified by: xiaofeng
# Last Modified time: 2018-04-12 12:03:05
'''
将数据集创建成npy的格式
'''
import numpy as np
import re, os
from IPython.display import display, Math, Latex, Image
from PIL import Image
import random
from tqdm import tqdm
import config as cfg
import datetime

VOCAB_PATH = cfg.VOCAB_PATH
FORMULAE_PATH = cfg.FORMULA_PATH
SET_LIST = cfg.SET_LIST

vocab = open(VOCAB_PATH).readlines()
formulae = open(FORMULAE_PATH, 'r').readlines()
char_to_idx = {x.split('\n')[0]: i for i, x in enumerate(vocab)}
char_to_idx['#UNK'] = len(char_to_idx)
char_to_idx['#START'] = len(char_to_idx)
char_to_idx['#END'] = len(char_to_idx)
idx_to_char = {y: x for x, y in char_to_idx.items()}
properties = {}
properties['vocab_size'] = len(vocab)
properties['vocab'] = vocab
properties['char_to_idx'] = char_to_idx
properties['idx_to_char'] = idx_to_char
print('saving properties!!')
np.save(cfg.DATA_ROOT + 'properties', properties)
print(len(char_to_idx))
print(char_to_idx)
for set in SET_LIST:
    print('current set is:', set)
    file_list = open(cfg.DATA_ROOT + set + "_filter.lst", 'r').readlines()
    set_list, missing = [], {}
    for i, line in enumerate(file_list):
        # file_list的形式为   7944775fc9.png 32771
        # form得到formulae对应的行数的位置
        form = formulae[int(line.split()[1])].strip().split()
        # out_form最开始的位置为['#START']-504,不存在于vocb中的字符为'#UNK'，结尾使用'#END'
        out_form = [char_to_idx['#START']]
        for char in form:
            try:
                out_form += [char_to_idx[char]]
            except:
                if char not in missing.keys():
                    print(char, " not found!")
                    missing[char] = 1
                else:
                    missing[char] += 1
                out_form += [char_to_idx['#UNK']]
        out_form += [char_to_idx['#END']]
        # set_list中存储为[图像名称，对应的图像的label]
        set_list.append([line.split()[0], out_form])
    buckets = {}
    file_not_found_count = 0
    file_not_found = []
    for img, label in tqdm(set_list):
        if os.path.exists(cfg.IMG_DATA_PATH + img):
            img_shp = Image.open(cfg.IMG_DATA_PATH + img).size
            try:
                buckets[img_shp] += [(img, label)]
            except:
                buckets[img_shp] = [(img, label)]
        else:
            file_not_found_count += 1
            file_not_found.append(img)
    Info_out = datetime.datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S') + '   ' + 'Num files found in %s set: %d/%d' % (
            set, len(set_list) - file_not_found_count,
            len(set_list)) + '\n\n' + 'Missing char:' + str(
                missing) + '\n\n' + 'Missing files:' + str(
                    file_not_found) + '\n\n\n'
    with open(cfg.DATA_ROOT + 'GenerateNPY.txt', 'a') as txt:
        txt.writelines(Info_out)
    txt.close()
    print(Info_out)
    # 保存成npy格式文件
    np.save(cfg.DATA_ROOT + set + '_buckets', buckets)
