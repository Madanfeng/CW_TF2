import tensorflow as tf
import numpy as np
import scipy.misc
import random
import time
import os

from inception_resnet_v2 import Dataset, InceptionResnetV2
from l2_attack import CarliniL2
from config import cfg

# NUM_LABELS = 20

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
 

if __name__ == "__main__":
    # road dataset and model
    data, model = Dataset(), InceptionResnetV2(model_path=cfg.MODEL_PATH)

    # Initialize CarliniL2
    attack = CarliniL2(model, batch_size=cfg.BATCH_SIZE, max_iterations=1000, confidence=cfg.CONFIDENCE)

    # inputs, targets = generate_data(data, samples=4, targeted=True, start=0)

    times = 0
    for i, (inputs, labs, targets) in enumerate(data):
        # 将targets转换成one-hot编码
        targets_tensor = tf.convert_to_tensor(targets, dtype=tf.int32)
        targets_tensor_onehot = tf.one_hot(targets_tensor, depth=data.class_num)

        timestart = time.time()
        advs = attack.attack_batch(inputs, targets_tensor_onehot)
        timeend = time.time()
        batch_time = timeend - timestart
        times += batch_time
        print("Took", batch_time, "seconds to run", data.batch_size, "samples.")

        for j, adv in enumerate(advs):
            pred = model.predict_softmax(adv)
            print('Classify and Save adversarial image %d_%d.' % (i, j))
            print("Classification:", np.argmax(pred), np.max(pred))
            scipy.misc.imsave('./output/adv'+str(cfg.CONFIDENCE)+'_'+str(i)+'_'+str(j)+'_'+str(np.argmax(pred))+'.jpg', adv)

            print("Total distortion:", np.sum((adv - inputs[j]) ** 2) ** .5)

    print("Took", times, "seconds to run all", data.num_samples, "samples.")
