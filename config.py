# 判断是否是在本地电脑运行
import os
A_NOTE = '路径存储信息'
LOCAL = True  # 是否本地运行
GENERATE = True  # 是否只是用数据库生成的数据进行训练
# 本地不使用数据库
if LOCAL and not GENERATE:
    DATA_ROOT = '/Users/xiaofeng/Code/Github/dataset/formula/original_data/prepared/'
    MODEL_SAVED = '/Users/xiaofeng/Code/Github/dataset/formula/original_data/model_saved/'
    PREDICT_PATH = '/Users/xiaofeng/Code/Github/dataset/formula/original_data/predict/'
# 远程不使用数据库
elif not LOCAL and not GENERATE:
    DATA_ROOT = '/home/xiaofeng/data/formula/prepared/'
    MODEL_SAVED = '/home/xiaofeng/data/formula/model_saved_remote/'
    PREDICT_PATH = '/home/xiaofeng/data/formula/predict_remote/'
# 本地使用数据库
elif LOCAL and GENERATE:
    DATA_ROOT = '/Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/'
    MODEL_SAVED = '/Users/xiaofeng/Code/Github/dataset/formula/generate/model_saved/'
    PREDICT_PATH = '/Users/xiaofeng/Code/Github/dataset/formula/generate/predict/'
# 远程使用数据库
else:
    DATA_ROOT = '/home/xiaofeng/data/formula/generate/prepared/'
    MODEL_SAVED = '/home/xiaofeng/data/formula/generate/model_saved_remote/'
    PREDICT_PATH = '/home/xiaofeng/data/formula/generate/predict_remote/'
if not os.path.exists(PREDICT_PATH):
    os.makedirs(PREDICT_PATH)
if not os.path.exists(MODEL_SAVED):
    os.makedirs(MODEL_SAVED)
SET_LIST = ['train', 'validate']
VOCAB_PATH = DATA_ROOT + 'latex_vocab.txt'
FORMULA_PATH = DATA_ROOT + 'formulas.norm.lst'
IMG_DATA_PATH = DATA_ROOT + 'images_processed/'
CHECKPOINT_PATH = MODEL_SAVED + 'ckpt/'
SUMMARY_PATH = MODEL_SAVED + 'log/'
PROPERTIES = DATA_ROOT + 'properties.npy'
LENGTH = 300
V_OUT = 3 + len(open(VOCAB_PATH).readlines())
SIZE_LIST = [(400, 160), (280, 40), (120, 50), (320, 40), (240, 50), (240, 40),
             (280, 50), (500, 100), (360, 40), (320, 50), (360, 100),
             (200, 50), (360, 60), (200, 40), (160, 40), (400, 50), (360, 50)]
RATIO = [size[0] / size[1] for size in SIZE_LIST]

BATCH_SIZE = 20
EPOCH_NUMS = 10000
LEARNING_RATE = 0.1
MIN_LEARNING_RATE = 0.001
DISPLAY_NUMS = 10
SAVED_NUMS = 100
