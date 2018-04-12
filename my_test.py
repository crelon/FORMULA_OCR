#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-09 12:18:15
# Last Modified by: xiaofeng
# Last Modified time: 2018-04-09 12:18:15

import numpy as np
from PIL import Image
import random
import tflib.ops
import os, requests, cv2


def formula_as_file(formula, file, negate=False):
    tfile = file
    if negate:
        tfile = 'tmp.png'
    r = requests.get(
        'http://latex.codecogs.com/png.latex?\dpi{300} \huge %s' % formula)
    f = open(tfile, 'wb')
    f.write(r.content)
    f.close()
    if negate:
        os.system(
            'convert tmp.png -channel RGB -negate -colorspace rgb %s' % file)


##保存图片
# ori = np.load('/Users/xiaofeng/Code/pred_imgs.npy')
# con_arr = np.squeeze(ori)
# img = Image.fromarray(con_arr)
# if img.mode != 'RGB':
#     img = img.convert('RGB')
# img.save('output.png')

ori = np.load('/Users/xiaofeng/Code/pred_latex.npy')
properties = np.load('properties.npy').tolist()
idx_to_chars = lambda Y: ' '.join(list(map(lambda x: properties['idx_to_char'][x], Y)))
str = idx_to_chars(ori[0, :])
# print('latex', str)

from IPython.lib.latextools import latex_to_png
from IPython.display import display, Image, Math
# import matplotlib
# import matplotlib.pyplot as plt
# plt.figure('1')
# str = r'\int _ { 1 } \! _ { ; } ( \pi ) \, \wp _ { \pm } ( X ) \partial k \leq }  { | { \bf ^ { \mu } } [ J _ { 1 } ( \Upsilon ) ] } ^ { \scriptscriptstyle } { } ^ { \scriptscriptstyle } { } ^ { \scriptscriptstyle } { - }  { | } { 2 \! ^ { 2 } | J _ { 2 } { ( N ) } ] ^ { \bot } { } _ { \, \alpha \kappa } }'
# str = r'T ^ { \scriptscriptstyle o v } ( X ) = \frac { 1 } { \pi \gamma ^ { \ast } \sqrt { c } } \int d \tau d ^ { \xi } \xi \; \dot { \pi } ^ { M } \dot { x } ^ { D } ( X ^ { \scriptscriptstyle m } - x ^ { M } )'
# str = r '\frac{2y+5}{2y-6}+\frac{1}{2}=\frac{4-3y}{4-2y}'
# str = r'\underbrace \! \! \! \! / p \, \! \! / \frac { \scriptsize } { 1 } \! \! \! / \, \frac { ( - h ) } { \scriptsize | } \! \! \! p ) } \, \, \, \frac { c } { \boldmath \stackrel { * } } \\ { \hline \! \! 5 } ) e _ { - 2 } \frac { - 2 } { \qquad m } _ { 2 }  { }  { } \\ { \scriptstyle \! \! \! p } \\ \end{array} _ { i } i'
# image = latex_to_png(s=str, backend='matplotlib', wrap=True)
# print(np.shape(image))
str = '\gamma ^ { \mu } ( - i \nabla _ { \mu } - \tilde { G } _ { \mu } ) { \bf \Psi } = 0 '
# displayPreds = lambda Y: display(Math(Y))
displayPreds = lambda Y: display(Math(Y.split('#END')[0]))
# displayPreds(str)
displayPreds('\frac{l}{n}\sigma^{\prime}(0),')
# formula_as_file(str, 'test_pred.jpg', False)

# original_test
set = 'train'
import random
# f = np.load('train_list_buckets.npy').tolist()
## function
'''
def predict(set='test', batch_size=1, visualize=True):
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
        if os.path.exists('./images_processed/' + f[start][0]):
            imgs.append(
                np.asarray(
                    Image.open('./images_processed/' + f[start][0]).convert(
                        'YCbCr'))[:, :, 0][:, :, None])
    # 转化成NCHW的形式
    imgs = np.asarray(imgs, dtype=np.float32).transpose(0, 3, 1, 2)
    inp_seqs = np.zeros((batch_size, 160)).astype('int32')
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

    for i in range(1, 160):
        inp_seqs[:, i] = sess.run(
            predictions, feed_dict={
                X: imgs,
                input_seqs: inp_seqs[:, :i]
            })
        print(i, inp_seqs)
        if visualize == True:
            # 对attention的值进行从大到小排列
            att = sorted(
                list(enumerate(tflib.ops.ctx_vector[-1].flatten())),
                key=lambda tup: tup[1],
                reverse=True)
            print('Attention', att)
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

    np.save('pred_imgs', imgs)
    np.save('pred_latex', inp_seqs)
    print("Saved npy files! Use Predict.ipynb to view results")
    return inp_seqs


predict()
'''