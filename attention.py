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
V = 502
NB_EPOCHS = 100000
H = 20
W = 50
PRECEPTION = 0.6
LEARNING_DECAY = 20000
LENGTH = cfg.LENGTH
LOCAL = True
IMG_PATH = cfg.IMG_DATA_PATH
PREDICT_PATH = cfg.PREDICT_PATH
ckpt_path = cfg.CHECKPOINT_PATH
summary_path = cfg.SUMMARY_PATH
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
if not os.path.exists(summary_path):
    os.makedirs(summary_path)
if not os.path.exists(PREDICT_PATH):
    os.makedirs(PREDICT_PATH)

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
def predict(set='test', batch_size=1, visualize=False):
    if visualize:
        assert (batch_size == 1), "Batch size should be 1 for visualize mode"
    import random
    # f = np.load('train_list_buckets.npy').tolist()
    f = np.load(set + '_buckets.npy').tolist()
    random_key = random.choice(list(f.keys()))
    #random_key = (160,40)
    f = f[random_key]
    imgs = []
    print("Image shape: ", random_key)
    while len(imgs) != batch_size:
        start = np.random.randint(0, len(f), 1)[0]
        print('Image name is :', f[start][0])
        if os.path.exists(IMG_PATH + f[start][0]):
            imgs.append(
                np.asarray(
                    Image.open(IMG_PATH + f[start][0]).convert('YCbCr'))
                [:, :, 0][:, :, None])
    print('Image shape is ', np.shape(imgs))
    imgs = np.asarray(imgs, dtype=np.float32).transpose(0, 3, 1, 2)
    print('image shape modified is ', np.shape(imgs))
    inp_seqs = np.zeros((batch_size, LENGTH)).astype('int32')
    inp_seqs[:, 0] = np.load('properties.npy').tolist()['char_to_idx'][
        '#START']
    tflib.ops.ctx_vector = []

    l_size = random_key[0] * 2
    r_size = random_key[1] * 2
    x = imgs[0][0]
    inp_image = Image.fromarray(imgs[0][0]).resize((l_size, r_size))
    l = int(np.ceil(random_key[1] / 8.))
    r = int(np.ceil(random_key[0] / 8.))
    properties = np.load('properties.npy').tolist()
    idx_to_chars = lambda Y: ' '.join(list(map(lambda x: properties['idx_to_char'][x],Y)))
    
    for i in range(1, LENGTH):
        inp_seqs[:, i] = sess.run(
            predictions, feed_dict={
                X: imgs,
                input_seqs: inp_seqs[:, :i]
            })
        # print(i, '\n', inp_seqs)
        print('Current step %d/%d' % (i, LENGTH))
        if visualize == True:
            # 对attention的值进行从大到小排列
            att = sorted(
                list(enumerate(tflib.ops.ctx_vector[-1].flatten())),
                key=lambda tup: tup[1],
                reverse=True)
            # print('Attention', att)
            idxs, att = zip(*att)
            j = 1
            while sum(att[:j]) < 0.9:
                j += 1
            positions = idxs[:j]
            print("Attention weights: ", att[:j])
            positions = [(pos / r, pos % r) for pos in positions]
            outarray = np.ones((l, r)) * 255.
            for loc in positions:
                outarray[int(loc[0]), int(loc[1])] = 0.
            out_image = Image.fromarray(outarray).resize((l_size, r_size),
                                                         Image.NEAREST)
            print("Latex sequence: ", idx_to_chars(inp_seqs[0, :i]))
            outp = Image.blend(
                inp_image.convert('RGBA'), out_image.convert('RGBA'), 0.5)
            outp.show(title=properties['idx_to_char'][inp_seqs[0, i]])
            # raw_input()
            time.sleep(3)
            os.system('pkill display')
    latex_str = idx_to_chars(inp_seqs[0, :])
    print(idx_to_chars(inp_seqs[0, :]).split('#END')[0].split('#START')[-1])
    np.save(PREDICT_PATH + 'pred_imgs_' + f[start][0].split('.')[0], imgs)
    np.save(PREDICT_PATH + 'pred_latex_' + f[start][0].split('.')[0], inp_seqs)
    print("Saved npy files! Use Predict.ipynb to view results")
    return inp_seqs


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
