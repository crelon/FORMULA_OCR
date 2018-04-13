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
'''

from pymongo import MongoClient
import re, os, sys, requests
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)
from PIL import Image
import tex2pix
import config as cfg
from sympy import preview
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
saved_path = '/Users/xiaofeng/Work_Guanghe/datasets/dataset'
if not os.path.exists(saved_path):
    os.makedirs(saved_path)

MIN_LENGTH = 20
MAX_LENGTH = 1024


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


def url_exist_or_not(content):
    read = content['body']
    result = re.findall("(?isu)(http\://[a-zA-Z0-9\.\?%+-/&\=\:]+)", read)
    return result


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
            if MAX_LENGTH > len(x.strip()) > MIN_LENGTH
        ]
        ret.extend(res)
    return ret


'''
连接数据库，进行待分析位置的定位
'''
# 创建MongoDB连接
client = MongoClient(host=Host, port=Port)
# print('which database does the client stored', client.database_names())

# 选择要连接的数据库名称
db = client[database_name]
# print('which collections does the database--%s stored' % db.name, db.collection_names())

# 选择当前数据库下的指定名称的collection
collections = db[Collection_name]

print('database name is %s ,collection name is %s' % (db.name,
                                                      collections.name))
'''
将id，body，url存储进txt中和dictionary中
'''
with open('formula.txt', 'w+') as txt:

    # 当前collection中包含的document数量
    length = collections.count()
    print('length', length)
    for content in collections.find():
        # result = url_exist_or_not(content)
        result = formula_exist_or_not(content)
        if result:
            count = 0
            id = content['_id']
            for res in result:
                # body = content['body']
                current_id = str(id) + '_' + str(count)
                formula = ''.join(res)
                dictionary['id'].append(id)
                # dictionary['body'].append(body)
                dictionary['formula'].append(result)
                saved_info = current_id + '   ' + str(formula) + '\n'
                # render = tex2pix.Renderer(formula, runbibtex=True)
                print(cfg.DATA_PULL + str(current_id) + '.jpg')
                formula_as_file(formula,
                                cfg.DATA_PULL + str(current_id) + '.jpg')
                txt.write(saved_info)
                count += 1
    print('Done')
txt.close()
print('total documents:', len(dictionary['id']))
# 总共存在的公式数量10587