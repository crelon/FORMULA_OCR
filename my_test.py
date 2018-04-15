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


str = r'\sqrt {a-5} -(3-b) \sqrt {b-3} =0'
formula_as_file(str, 'test_pred.jpg', False)

##保存图片
# ori = np.load('/Users/xiaofeng/Code/pred_imgs.npy')
# con_arr = np.squeeze(ori)
# img = Image.fromarray(con_arr)
# if img.mode != 'RGB':
#     img = img.convert('RGB')
# img.save('output.png')

# ori = np.load('/Users/xiaofeng/Code/pred_latex.npy')
# properties = np.load('properties.npy').tolist()
# idx_to_chars = lambda Y: ' '.join(list(map(lambda x: properties['idx_to_char'][x], Y)))
# str = idx_to_chars(ori[0, :])
# print('latex', str)
