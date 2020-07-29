from easydict import EasyDict as edict

CLASS_NAME = ['notkonw1', 'notkonw2', 'das', 'agka', 'erg',
              'feet', 'in', 'tev', 'inv', 'gy',
              'maa', 'mif', 'mage', 'ked', 'nmal',
              'pog', 'sgfx', 'sedy', 'sy', 'dsm']

__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

__C.ANNOT_PATH                = './data/dataset/dataset.txt'
__C.MODEL_PATH                = './model/model.stage2.05-0.8851.v2_320_tf2.hdf5'

__C.BATCH_SIZE                = 2
__C.IMAGE_SIZE                = 320
__C.NUM_CHANNELS              = 3
__C.CLASS_NAME                = CLASS_NAME

__C.TARGET                    = 14
__C.CONFIDENCE                = 40
