# -*- coding: utf-8 -*-
_Author_ = 'xiaofeng'
'''
连接MongoDB数据库，找到存在http网址的字符串；
将url保存并下载到本地，保存位置为'./data/image/'
图片明明规则：5a45403a8223977701b0aa6a_3.5-Y-A3-2-2-1.png
            object id_src.split('/')[-1]
可以从图片的名称中，方便的找到该图片对应的数据库中的id以及图片的url地址信息
'''

from pymongo import MongoClient
import re, os
import requests
from PIL import Image
from io import BytesIO
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
dictionary['url'] = []
saved_path = '/Users/xiaofeng/Work_Guanghe/datasets/dataset'
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
'''
判断body中是否存在http网址，并将该网址保存在list中
'''


def url_exist_or_not(content):
    read = content['body']
    result = re.findall("(?isu)(http\://[a-zA-Z0-9\.\?%+-/&\=\:]+)", read)
    return result


def formula_exist_or_not(content):
    read = content['body']
    result = re.findall('.*\$|\$$(.*)\$|\$$.*', read)
    return result


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
with open('url.txt', 'w+') as txt:
    count = 0
    # 当前collection中包含的document数量
    length = collections.count()
    for content in collections.find():
        print('body 的内容', content)
        # result = url_exist_or_not(content)
        result = formula_exist_or_not(content)
        print(result)
        if result:
            id = str(content['_id'])
            body = content['body']
            url = '\r\n'.join(result)
            dictionary['id'].append(id)
            dictionary['body'].append(body)
            dictionary['url'].append(result)
            saved_info = str(id) + ' ' + str(body) + '' + str(url) + '\n'
            # print(saved_info)
            saved_info.encode('utf-8', 'ignore').decode('utf-8')
            # txt.writelines(saved_info)
        count += 1
        # print('{}/{}...'.format(count, length))
    print('Done')
print('total documents:', len(dictionary['id']))
'''
根据得到的url进行数据下载，并保存在本地；
在确定网络架构之后，可能没必要进行图片保存在本地，直接使用url就可以
本次得到的所有图片总共2819张，对应着题目的数量为：2727
'''
####### 下载网址图片
image_nums = len(dictionary['url'])
image_count = 1
for i in range(image_nums):
    for image_src in dictionary['url'][i]:
        image_name = image_src.split('/')[-1]
        response = requests.get(image_src)
        image = Image.open(BytesIO(response.content))
        saved_name = os.path.join(saved_path, image_name)
        # saved_name = saved_path + dictionary['id'][i] + '_' + image_name
        image.save(fp=saved_name)
        image_count += 1
        print('{}...'.format(image_count))
print('done!', 'total image num is:', image_count)
