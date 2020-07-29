from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import random
import scipy.misc

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from config import cfg

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# # BATCH_SIZE   = 1
IMAGE_SIZE   = 320
NUM_CHANNELS = 3
NUM_LABELS   = 20
#
DATA_PATH = './data/imgs/'
# # MODEL_PATH = './model/model.stage4.04-0.9697.v2_320_tf2.hdf5'
# MODEL_PATH = './model/model.stage2.05-0.8851.v2_320_tf2.hdf5'
#
# CLASS_NAME = []


class InceptionResnetV2:
    # load_model from trained .hdf5
    def __init__(self, model_path):
        self.image_size = cfg.IMAGE_SIZE
        self.num_channels = cfg.NUM_CHANNELS
        self.num_labels = NUM_LABELS
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model_logits = Model(inputs=self.model.input, outputs=self.model.layers[-2].output)
        self.model.summary()

    def predict_logits(self, imgs):
        # imgs.shape = [batch_size, image_size, image_size, num_channels]
        return self.model_logits(imgs)

    def predict_softmax(self, imgs):
        # imgs.shape = [batch_size, image_size, image_size, num_channels]
        return self.model(imgs)


def readimg(ff):
    # ff : LabelID_ImgName.jpg
    # return [img, labelID]
    f = DATA_PATH + ff
    img = np.array(scipy.misc.imresize(scipy.misc.imread(f), (IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255 - .5
    if img.shape != (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS):
        return None
    return [img, int(ff.split("_")[0])]


class MyData:
    # demo
    def __init__(self):
        from multiprocessing import Pool
        pool = Pool(8)
        r = pool.map(readimg, os.listdir(DATA_PATH))
        r = [x for x in r if x != None]
        test_data, test_labels = zip(*r)
        self.test_data = np.array(test_data)
        # one-hot
        self.test_labels = np.zeros((len(test_labels), NUM_LABELS))
        self.test_labels[np.arange(len(test_labels)), test_labels] = 1


class Dataset:
    # dataset with iter
    # using dataset to attack model or evaluate model.
    # 目前的版本是从 annotation 文件中读取图片路径和标签 以此读取图片
    # batch_images, batch_labels, batch_target
    # 注意：其中图片已经resize,经过归一化处理 0.5~-0.5， 标签不是one-hot编码
    def __init__(self):
        self.target       = cfg.TARGET  # 目标类，本代码统一将不同图片攻击成某一设定类别
        self.annot_path   = cfg.ANNOT_PATH

        self.batch_size   = cfg.BATCH_SIZE
        self.image_size   = cfg.IMAGE_SIZE
        self.num_channels = cfg.NUM_CHANNELS
        self.class_name   = cfg.CLASS_NAME
        self.class_num    = len(self.class_name)

        self.annotations  = self.load_annotations()
        self.num_samples  = len(self.annotations)
        self.num_batchs   = int(np.ceil(self.num_samples / self.batch_size))  # 一共需要分为多少批次
        self.batch_count  = 0   # 记录已有多少个batch

        if self.target < 0 or self.target >= self.class_num:
            raise KeyError("The target had an error in the value.")

    def load_annotations(self):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        # np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):

        with tf.device('/cpu:0'):
            batch_images = np.zeros((self.batch_size, self.image_size, self.image_size, self.num_channels),
                                    dtype=np.float32)
            batch_labels = np.ones(self.batch_size, dtype=np.float32)
            batch_target = np.ones(self.batch_size, dtype=np.float32) * self.target

            num = 0   # 记录当前batch中样本的个数
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]
                    image, label = self.parse_annotation(annotation)

                    batch_images[num, :, :, :] = image
                    batch_labels[num] *= label

                    if int(label) == int(self.target):
                        # 如果图片标签和目标类一样，则随机选取一个目标类
                        target_pool = [i for i in range(self.class_num)]
                        target_pool.pop(self.target)
                        batch_target[num, :] = random.sample(target_pool, 1)

                    num += 1

                self.batch_count += 1
                return batch_images, batch_labels, batch_target
            else:
                self.batch_count = 0
                # np.random.shuffle(self.annotations)
                raise StopIteration

    def __len__(self):
        return self.num_batchs

    def parse_annotation(self, annotation):
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (self.image_size, self.image_size))
        image_resized = image_resized / 255. - .5

        label = int(line[1])
        return image_resized, label


def main():
    # # read and precess image
    # image_path = './output/adv3+14.png'
    # # image_path = './imgs/665bc90eb487cb0760d40854b6dfe9cd.jpg'
    # # image = scipy.misc.imresize(scipy.misc.imread(image_path), (IMAGE_SIZE, IMAGE_SIZE))
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image[np.newaxis, :] / 255. - 0.5
    #
    # # road InceptionResnetV2 model
    # model = InceptionResnetV2()
    # pred = model.predict_softmax(image)
    # pred_id = np.argmax(pred)
    # print("The " + image_path.split('/')[-1] + " is classified as: ", pred_id, pred.numpy())

    # road InceptionResnetV2 model
    model = InceptionResnetV2(model_path=cfg.MODEL_PATH)

    # road dataset
    data = Dataset()

    # classification
    true_num = 0
    target_num = 0
    num = data.num_samples
    for i, (images, labs, targets) in enumerate(data):
        preds = model.predict_softmax(images)
        for j, pred in enumerate(preds):
            pred_id = np.argmax(pred)
            print("The " + str(i)+'_'+str(j) + " is classified as:", pred_id)
            print(pred.numpy())
            if pred_id == labs[j]:
                true_num += 1
            elif pred_id == targets[j]:
                target_num += 1

    print("Classification Success Rate: ", true_num / num)
    print("Attack Success Rate: ", target_num / num)
    print("Done.")


if __name__ == '__main__':
    main()
