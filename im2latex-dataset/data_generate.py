#!/usr/bin/env python
# _Author_: xiaofeng
# Date: 2018-04-13 17:20:01
# Last Modified by: xiaofeng
# Last Modified time: 2018-04-13 17:20:01
# -*- coding: utf-8 -*-
'''
连接MongoDB数据库，找到存在http网址的字符串；
将url保存并下载到本地，保存位置为'./data/image/'
图片明明规则：5a45403a8223977701b0aa6a_3.5-Y-A3-2-2-1.png
            object id_src.split('/')[-1]
可以从图片的名称中，方便的找到该图片对应的数据库中的id以及图片的url地址信息
    1.将数据库中的body进行正则匹配
    2.将匹配出来的latex格式进行txt格式存储
    3.使用latex生成pdf，在转换成png格式
'''

from pymongo import MongoClient
import re, os, sys, requests, glob
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)
from subprocess import call
import hashlib
from PIL import Image
import random
import tex2pix
import config as cfg
from multiprocessing import Pool
'''
预设地址，端口，数据库名称，collection名称
'''
Host = '10.8.8.71'
Port = 27017
database_name = 'knowledge_graph'
Collection_name = 'problems_info'
'''
预设进行数据存储的dict
'''
dictionary = {}
dictionary['id'] = []
dictionary['body'] = []
dictionary['formula'] = []

MIN_LENGTH = 10
MAX_LENGTH = 500
MAX_NUMBER = 150 * 1000
THREADS = 3
TRAIN_PERSP = 0.8
DEVNULL = open(os.devnull, "w")
# Running a thread pool masks debug output. Set DEBUG to 1 to run
# formulas over images sequentially to see debug errors more clearly
DEBUG = False
FORMULA_TXT = './generate/formula.txt'
NAME_TXT = './generate/name.txt'
DATASET_FILE = "./generate/im2latex.lst"
NEW_FORMULA_FILE = "./generate/im2latex_formulas.lst"
TRAIN_LIST = './generate/train.list'
VALIDATE_LIST = './generate/validate.list'
IMAGE_DIR = '/Users/xiaofeng/Code/Github/dataset/formula/data_formula'
BASIC_SKELETON = r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{amsmath}
\begin{document}

\begin{displaymath}
%s
\end{displaymath}

\end{document}
"""

RENDERING_SETUPS = {
    'basic': [
        BASIC_SKELETON, "convert -density 200 -quality 100 %s.pdf %s.png",
        lambda filename: os.path.isfile(filename + ".png")
    ]
}


def remove_temp_files(name):
    """ Removes .aux, .log, .pdf and .tex files for name """
    try:
        os.remove(name + ".aux")
        os.remove(name + ".log")
        os.remove(name + ".pdf")
        os.remove(name + ".tex")
    except:
        pass


# 通过使用网址‘http://latex.codecogs.com’来生成img，会存在很多无法识别的情况
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


# 进行网址信息的正则匹配
def url_exist_or_not(content):
    read = content['body']
    result = re.findall("(?isu)(http\://[a-zA-Z0-9\.\?%+-/&\=\:]+)", read)
    return result


# 进行公式的正则匹配
def formula_exist_or_not(content):
    read = content['body']
    pattern = [
        r"\\begin\{equation\}(.*?)\\end\{equation\}", r"\$\$(.*?)\$\$",
        r"\$(.*?)\$", r"\\\[(.*?)\\\]", r"\\\((.*?)\\\)"
    ]
    ret = []
    for pat in pattern:
        res = re.findall(pat, read, re.DOTALL)
        res = [
            x.strip().replace('\n', '').replace('\r', '') for x in res
            if MAX_LENGTH > len(list(set(x.strip()))) > MIN_LENGTH
        ]
        ret.extend(res)
    return ret


# 连接数据库，进行待分析位置的定位
# 创建MongoDB连接
client = MongoClient(host=Host, port=Port)
# 选择要连接的数据库名称
db = client[database_name]
# 选择当前数据库下的指定名称的collection
collections = db[Collection_name]

print('database name is %s ,collection name is %s' % (db.name,
                                                      collections.name))
'''
将id，formula存储进txt中和dictionary中
'''


# 生成公式txt
def formula_txt():
    # 当前collection中包含的document数量
    with open(FORMULA_TXT, 'w') as f:
        with open(NAME_TXT, 'w') as g:
            length = collections.count()
            print('total length', length)
            number = 1
            for content in collections.find():
                # result = url_exist_or_not(content)
                result = formula_exist_or_not(content)
                if result:
                    count = 0
                    id = content['_id']
                    for res in result:
                        current_id = str(id) + '_' + str(count)
                        formula = ''.join(res)
                        formu_info = str(formula) + '\n'
                        name_info = current_id + '\n'
                        f.write(formu_info)
                        g.write(name_info)
                        count += 1
                        number += 1
    print('生成的公式数量', number)


def formula_to_image(formula):
    """ Turns given formula into images based on RENDERING_SETUPS
    returns list of lists [[image_name, rendering_setup], ...], one list for
    each rendering.
    Return None if couldn't render the formula"""
    formula = formula.strip("%")
    name = hashlib.sha1(formula.encode('utf-8')).hexdigest()[:-1]
    ret = []
    skiping = []
    for rend_name, rend_setup in RENDERING_SETUPS.items():
        full_path = name + "_" + rend_name
        if len(rend_setup) > 2 and rend_setup[2](full_path):
            print("Skipping, already done: " + full_path)
            ret.append([full_path, rend_name])
            continue
        # Create latex source
        latex = rend_setup[0] % formula
        # Write latex source
        with open(full_path + ".tex", "w") as f:
            f.write(latex)

        # Call pdflatex to turn .tex into .pdf
        code = call(
            [
                "pdflatex", '-interaction=nonstopmode', '-halt-on-error',
                full_path + ".tex"
            ],
            stdout=DEVNULL,
            stderr=DEVNULL)
        if code != 0:
            os.system("rm -rf " + full_path + "*")
            return None

        # Turn .pdf to .png
        # Handles variable number of places to insert path.
        # i.e. "%s.tex" vs "%s.pdf %s.png"
        full_path_strings = rend_setup[1].count("%") * (full_path, )

        code = call(
            (rend_setup[1] % full_path_strings).split(" "),
            stdout=DEVNULL,
            stderr=DEVNULL)
        #Remove files
        try:
            remove_temp_files(full_path)
        except Exception as e:
            # try-except in case one of the previous scripts removes these files
            # already
            return None

        # Detect of convert created multiple images -> multi-page PDF
        resulted_images = glob.glob(full_path + "-*")
        if code != 0:
            # Error during rendering, remove files and return None
            os.system("rm -rf " + full_path + "*")
            return None
        elif len(resulted_images) > 1:
            # We have multiple images for same formula
            # Discard result and remove files
            for filename in resulted_images:
                os.system("rm -rf " + filename + "*")
            return None
        else:
            ret.append([full_path, rend_name])

    return ret


def generate_formula_lst_img():
    if not os.path.exists(FORMULA_TXT) and not os.path.exists(NAME_TXT):
        formula_txt()
        formulas = open(FORMULA_TXT).read().split('\n')
    else:
        formulas = open(FORMULA_TXT).read().split('\n')
    try:
        os.mkdir(IMAGE_DIR)
    except OSError as e:
        pass  #except because throws OSError if dir exists
    print("Turning formulas into images...")
    # Change to image dir because textogif doesn't seem to work otherwise...
    oldcwd = os.getcwd()
    # Check we are not in image dir yet (avoid exceptions)
    if not IMAGE_DIR in os.getcwd():
        os.chdir(IMAGE_DIR)

    names = None

    if DEBUG:
        names = [formula_to_image(formula) for formula in formulas]
    else:
        pool = Pool(THREADS)
        names = list(pool.imap(formula_to_image, formulas))
    # 切换到到当前路径
    os.chdir(oldcwd)

    zipped = list(zip(formulas, names))

    new_dataset_lines = []
    new_formulas = []
    ctr = 0
    for formula in zipped:
        if formula[1] is None:
            continue
        for rendering_setup in formula[1]:
            new_dataset_lines.append(
                str(ctr) + " " + " ".join(rendering_setup))
        new_formulas.append(formula[0])
        ctr += 1
    print('total', ctr)
    with open(NEW_FORMULA_FILE, "w") as f:
        f.write("\n".join(new_formulas))

    with open(DATASET_FILE, "w") as f:
        f.write("\n".join(new_dataset_lines))


#
def generate_train_test_batch():
    if not os.path.exists(DATASET_FILE):
        generate_formula_lst_img()
        formulas = open(DATASET_FILE).read().split('\n')
    else:
        formulas = open(DATASET_FILE).read().split('\n')
    total_length = len(formulas)
    train_length = int(TRAIN_PERSP * total_length)
    random.shuffle(formulas)
    train = formulas[:train_length]
    valid = formulas[train_length:]
    print(len(train), len(valid))
    print('creating train an test lis')
    with open(TRAIN_LIST, 'w') as tr:
        tr.write('\n'.join(train))
    with open(VALIDATE_LIST, 'w') as va:
        va.write('\n'.join(valid))


if __name__ == '__main__':
    generate_train_test_batch()
