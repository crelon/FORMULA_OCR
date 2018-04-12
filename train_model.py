#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-01 12:12:42
# Last Modified by: xiaofeng
# Last Modified time: 2018-04-01 12:12:42

from PIL import Image
import tensorflow as tf
import tflib
import tflib.ops
import tflib.network
from tqdm import tqdm
import numpy as np
import data_loaders
import time
import os
import config as cfg

BATCH_SIZE = 20
EMB_DIM = 80
ENC_DIM = 256
DEC_DIM = ENC_DIM * 2
NUM_FEATS_START = 64
D = NUM_FEATS_START * 8
V = 506  # vocab size
NB_EPOCHS = 100000
H = 20
W = 50
PRECEPTION = 0.6
LEARNING_DECAY = 20000
LENGTH = cfg.LENGTH
LOCAL = True
IMG_PATH = cfg.IMG_DATA_PATH
PREDICT_PATH = cfg.PREDICT_PATH
PROPERTIES = cfg.PROPERTIES
ckpt_path = cfg.CHECKPOINT_PATH
summary_path = cfg.SUMMARY_PATH
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
if not os.path.exists(summary_path):
    os.makedirs(summary_path)
if not os.path.exists(PREDICT_PATH):
    os.makedirs(PREDICT_PATH)

with open('config.txt', 'w+') as f:
    cfg_dict = cfg.__dict__
    for key in sorted(cfg_dict.keys()):
        if key[0].isupper():
            cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
            f.write(cfg_str)
    f.close()

# with tf.device("/cpu:0"):
#     custom_runner = data_loaders.CustomRunner()
#     X, seqs, mask = custom_runner.get_inputs()
#
# print X,seqs
X = tf.placeholder(shape=(None, None, None, None), dtype=tf.float32)
mask = tf.placeholder(shape=(None, None), dtype=tf.int32)
seqs = tf.placeholder(shape=(None, None), dtype=tf.int32)
learn_rate = tf.placeholder(tf.float32)
input_seqs = seqs[:, :-1]
target_seqs = seqs[:, 1:]
emb_seqs = tflib.ops.Embedding('Embedding', V, EMB_DIM, input_seqs)

ctx = tflib.network.im2latex_cnn(X, NUM_FEATS_START, True)
# out,state = tflib.ops.FreeRunIm2LatexAttention('AttLSTM',ctx,emb_seqs,EMB_DIM,ENC_DIM,DEC_DIM,D,H,W)
out, state = tflib.ops.im2latexAttention('AttLSTM', emb_seqs, ctx, EMB_DIM,
                                         ENC_DIM, DEC_DIM, D, H, W)
logits = tflib.ops.Linear('MLP.1', out, DEC_DIM, V)
# logits = tflib.ops.Linear('.output_t', out, DEC_DIM, V)

predictions = tf.argmax(tf.nn.softmax(logits[:, -1]), axis=1)
loss = tf.reshape(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(logits, [-1, V]),
        labels=tf.reshape(seqs[:, 1:], [-1])), [tf.shape(X)[0], -1])
# add paragraph ⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️
output = tf.reshape(logits, [-1, V])
output_index = tf.to_int32(tf.argmax(output, 1))
true_labels = tf.reshape(seqs[:, 1:], [-1])
correct_prediction = tf.equal(output_index, true_labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# ⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️
mask_mult = tf.to_float(mask[:, 1:])
loss = tf.reduce_sum(loss * mask_mult) / tf.reduce_sum(mask_mult)
#train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learn_rate)
gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gvs]
train_step = optimizer.apply_gradients(capped_gvs)
#### summary
tf.summary.scalar('model_loss', loss)
tf.summary.scalar('model_accuracy', accuracy)
gradient_norms = [tf.norm(grad) for grad, var in gvs]
tf.summary.histogram('gradient_norm', gradient_norms)
tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
merged = tf.summary.merge_all()


## function to predict the latex
def score(set='valid', batch_size=32):
    score_itr = data_loaders.data_iterator(set, batch_size)
    losses = []
    start = time.time()
    for score_imgs, score_seqs, score_mask in score_itr:
        _loss = sess.run(
            loss,
            feed_dict={
                X: score_imgs,
                seqs: score_seqs,
                mask: score_mask
            })
        losses.append(_loss)
    set_loss = np.mean(losses)

    perp = np.mean(list(map(lambda x: np.power(np.e, x), losses)))

    print("\tMean  Loss: %s " % set_loss)
    print("\tTotal Time: {} ".format(time.time() - start))
    print("\tMean Perplexity: %s " % perp)
    return set_loss, perp


config = tf.ConfigProto(intra_op_parallelism_threads=8)
config.gpu_options.per_process_gpu_memory_fraction = PRECEPTION
sess = tf.Session(config=config)
# init = tf.global_variables_initializer()
init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
# init = tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=0.5)
saver_path = os.path.join(ckpt_path, 'weights_best.ckpt')
file_list = os.listdir(ckpt_path)
if file_list:
    for i in file_list:
        if i == 'checkpoint':
            print('Restore the weight files form:', ckpt_path)
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
suammry_writer = tf.summary.FileWriter(
    summary_path, flush_secs=60, graph=sess.graph)

# saver.restore(sess,'./weights_best.ckpt')
## start the tensorflow QueueRunner's
coord = tf.train.Coordinator()
thread = tf.train.start_queue_runners(sess=sess, coord=coord)
## start our custom queue runner's threads
# custom_runner.start_threads(sess)
## 进行预测
predict()

losses = []
times = []
print("Compiled Train function!")
## Test is train func runs
# train_fn(np.random.randn(32,1,128,256).astype('float32'),np.random.randint(0,107,(32,50)).astype('int32'),np.random.randint(0,2,(32,50)).astype('int32'), np.zeros((32,1024)).astype('float32'))
i = 0
lr = 0.1
iter = 0
best_perp = np.finfo(np.float32).max
for i in range(i, NB_EPOCHS):
    print('best_perp', best_perp)
    print('Learning rate is :', lr)
    costs = []
    times = []
    pred = []
    itr = data_loaders.data_iterator('train', BATCH_SIZE)
    for train_img, train_seq, train_mask in itr:
        iter += 1
        start = time.time()
        _, _loss, _acc, summary = sess.run(
            [train_step, loss, accuracy, merged],
            feed_dict={
                X: train_img,
                seqs: train_seq,
                mask: train_mask,
                learn_rate: lr
            })
        # _ , _loss = sess.run([train_step,loss],feed_dict={X:train_img,seqs:train_seq,mask:train_mask})
        times.append(time.time() - start)
        costs.append(_loss)
        pred.append(_acc)
        if iter % 100 == 0:
            print("Iter: %d (Epoch %d--%d)" % (iter, i + 1, NB_EPOCHS))
            print("\tMean cost: ", np.mean(costs))
            print("\tMean prediction: ", np.mean(pred))
            print("\tMean time: ", np.mean(times))
            print('\tSaveing summary to the path:', summary_path)
            print('\tSaveing model to the path:', saver_path)
            suammry_writer.add_summary(summary, global_step=iter * i + iter)
            saver.save(sess, saver_path)

    print("\n\nEpoch %d Completed!" % (i + 1))
    print("\tMean train cost: ", np.mean(costs))
    print("\tMean train perplexity: ",
          np.mean(list(map(lambda x: np.power(np.e, x), costs))))
    print("\tMean time: ", np.mean(times))
    val_loss, val_perp = score('valid', BATCH_SIZE)
    print("\tMean val cost: ", val_loss)
    print("\tMean val perplexity: ", val_perp)
    if val_perp < best_perp:
        best_perp = val_perp
        saver.save(sess, saver_path)
        print("\tBest Perplexity Till Now! Saving state!")
    if iter > 0 and iter / LEARNING_DECAY == 0:
        lr = lr * 0.5
    print("\n\n")
coord.request_stop()
coord.join(thread)

#sess.run([train_step,loss],feed_dict={X:np.random.randn(32,1,256,512),seqs:np.random.randint(0,107,(32,40)),mask:np.random.randint(0,2,(32,40))})
