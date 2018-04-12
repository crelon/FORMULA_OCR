# 判断是否是在本地电脑运行
import os
LOCAL = True
if LOCAL:
    DATA_ROOT = '/Users/xiaofeng/Code/Github/dataset/formula/prepared/'
    MODEL_SAVED = '/Users/xiaofeng/Code/Github/dataset/formula/model_saved/'
    PREDICT_PATH = '/Users/xiaofeng/Code/Github/dataset/formula/predict/'

else:
    DATA_ROOT = os.path.abspath('.') + '/'
    MODEL_SAVED = os.path.join(os.path.abspath('.'), 'model_saved')
    PREDICT_PATH = os.path.join(os.path.abspath('.'), 'predict')
if not os.path.exists(PREDICT_PATH):
    os.makedirs(PREDICT_PATH)
SET_LIST = ['train', 'test', 'validate']
VOCAB_PATH = DATA_ROOT + 'latex_vocab.txt'
FORMULA_PATH = DATA_ROOT + 'formulas.norm.lst'
IMG_DATA_PATH = DATA_ROOT + 'images_processed/'
CHECKPOINT_PATH = MODEL_SAVED + 'ckpt/'
SUMMARY_PATH = MODEL_SAVED + 'log/'
LENGTH = 300
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
