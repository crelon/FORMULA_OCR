#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-10 12:08:06
# Last Modified by: xiaofeng
# Last Modified time: 2018-04-10 12:08:06
'''
forked from predict.py
'''
from PIL import Image
import tensorflow as tf
import tflib
import tflib.ops
import tflib.network
from tqdm import tqdm
import numpy as np
import data_loaders
# import time
import os
import json
import cv2
import sys
# import pyperclip
sys.path.append('./im2markup/scripts/utils')
# from image_utils import *
# import subprocess
# import glob
from IPython.display import display, Math, Latex
from IPython.display import Image as Img
from io import StringIO
import config as cfg
# import re

BATCH_SIZE = 1  # 预测时，batch_size 必须为1
EMB_DIM = 80
ENC_DIM = 256
DEC_DIM = ENC_DIM * 2
NUM_FEATS_START = 64
D = NUM_FEATS_START * 8
# V = 502
V = 502
NB_EPOCHS = 50
H = 20
W = 50
PRECEPTION = 0.6
LENGTH = cfg.LENGTH
RATIO = cfg.RATIO
SIZE = cfg.SIZE_LIST
Saved_path = cfg.MODEL_SAVED
# 待验证的文件的存储位置
# 模型存储路径
ckpt_path = os.path.join(Saved_path, 'ckpt')
summary_path = os.path.join(Saved_path, 'log')
# build the model
X = tf.placeholder(shape=(None, None, None, None), dtype=tf.float32)
mask = tf.placeholder(shape=(None, None), dtype=tf.int32)
seqs = tf.placeholder(shape=(None, None), dtype=tf.int32)
learn_rate = tf.placeholder(tf.float32)
input_seqs = seqs[:, :-1]
target_seqs = seqs[:, 1:]
emb_seqs = tflib.ops.Embedding('Embedding', V, EMB_DIM, input_seqs)

ctx = tflib.network.im2latex_cnn(X, NUM_FEATS_START, True)
out, state = tflib.ops.im2latexAttention('AttLSTM', emb_seqs, ctx, EMB_DIM,
                                         ENC_DIM, DEC_DIM, D, H, W)
logits = tflib.ops.Linear('MLP.1', out, DEC_DIM, V)
# 设置输出的阈值，查看输出的结果
out_predict = tf.nn.softmax(logits[:, -1], axis=1)

# predictions = tf.argmax(tf.nn.softmax(logits[:, -1]), axis=1)
predictions = tf.argmax(
    tf.nn.softmax_cross_entropy_with_logits(logits[:, -1]), axis=1)

loss = tf.reshape(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(logits, [-1, V]),
        labels=tf.reshape(seqs[:, 1:], [-1])), [tf.shape(X)[0], -1])
mask_mult = tf.to_float(mask[:, 1:])
loss = tf.reduce_sum(loss * mask_mult) / tf.reduce_sum(mask_mult)
optimizer = tf.train.GradientDescentOptimizer(learn_rate)
gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gvs]
train_step = optimizer.apply_gradients(capped_gvs)
# 进行config配置
config = tf.ConfigProto(intra_op_parallelism_threads=8)
config.gpu_options.per_process_gpu_memory_fraction = PRECEPTION
sess = tf.Session(config=config)
init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
sess.run(init)
saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=0.5)
saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
coord = tf.train.Coordinator()
thread = tf.train.start_queue_runners(sess=sess, coord=coord)

# 加载存储的符号文件并转化成本list形式
properties = np.load('properties.npy').tolist()
print('properties', properties)


def predict_img_latex(file):
    # 将图片转成数组形式
    imgs = Image.open(file)
    width, height = imgs.size
    print('Image ori height, width is :', height, width)
    ratio = width / height
    ratio_diffreent = [np.abs(i - ratio) for i in RATIO]
    size_index = ratio_diffreent.index(min(ratio_diffreent))
    imgs = imgs.resize(SIZE[size_index], Image.LANCZOS)
    width, height = imgs.size
    print('Image new height, width is  :', height, width)
    print('Save the resized image....')
    file_name = file.split('/')[-1].split('.')[0]
    imgs.save(cfg.PREDICT_PATH + file_name + '_resize.' + file.split('.')[-1])
    imgs = np.asarray(imgs.convert('YCbCr'))[:, :, 0][None, None, :]
    print('shape', np.shape(imgs))
    # 预测输出字符串的长度
    char_length = int(width / 2)
    # imgs为NCHW形式，将其转化成NHWC形式
    imgs = np.asarray(imgs, dtype=np.float32).transpose(0, 2, 3, 1)
    inp_seqs = np.zeros((BATCH_SIZE, char_length)).astype('int32')
    inp_seqs[0, :] = properties['char_to_idx']['#START']
    tflib.ops.ctx_vector = []
    displayPreds = lambda Y: display(Math(Y.split('#END')[0]))
    idx_to_chars = lambda Y: ' '.join(list(map(lambda x: properties['idx_to_char'][x],Y)))
    # predict the latex
    for i in range(1, char_length):
        inp_seqs[:, i], pre = sess.run(
            (predictions, out_predict),
            feed_dict={
                X: imgs,
                input_seqs: inp_seqs[:, :i]
            })

        print('cureent position index is %d-->>%d' % (i, LENGTH))
        predict_want = pre[0]
        predict_want = sorted(predict_want, reverse=True)
        print('total is :', sum(predict_want))
        print('prediction', predict_want[:5])

    str_ori = idx_to_chars(inp_seqs[0, :])
    print('original latex is ：', str_ori)
    str = idx_to_chars(inp_seqs.flatten().tolist()).split('#END')[0].replace(
        '\left', '').replace('\\right', '').replace('&', '')
    print('typed latex is ：', str)
    '''
    # pyperclip.copy('$' + str + '$')
    def showarray(a, fmt='png'):
        a = np.uint8(a)
        f = StringIO()
        print('file name is ', f)
        Image.fromarray(a).save(f, fmt)
        # display(Img(data=f.getvalue()))

    # 结果输出
    preds_chars = idx_to_chars(str[1:]).replace('$', '')
    print('Original input image name is ', file)
    showarray(imgs[0])
    print("Predicted Latex")
    print(preds_chars.split('#END')[0])
    print("\nRendering the predicted latex")
    displayPreds(preds_chars)
    print("\n")
    '''


predict_img_latex('./1.jpg')
# predict_img_latex('./images_processed/570d6766c0.png')
latex