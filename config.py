from easydict import EasyDict as edict

CLASS_NAME = ['notkonw1', 'notkonw2', 'baby_naked', 'child_naked', 'exposed',
              'feet', 'intimate_nannan', 'intimate_nannv', 'intimate_nvnv', 'leggy',
              'man_half_bareness', 'midriff', 'mild_cleavage', 'naked', 'normal',
              'porn', 'sex', 'sex_toy', 'sexy', 'sm']

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
