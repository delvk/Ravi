import numpy as np
import cv2
import tensorflow.compat.v1 as tf
import os
import random
from random import *
import math
import sys
import copy
from glob import glob

# from .heatmap import *
import pandas as pd
from PIL import Image


class MCNN:
    def __init__(self, dataset, learning_rate):
        self.dataset = dataset
        self.LEARNING_RATE = learning_rate
        self.name = "MCNN"
        self.build()

    def build(self):
        # Build model
        self.input_image = tf.placeholder(tf.float32, [None, None, None, 1])
        self.ground_truth = tf.placeholder(tf.float32, [None, None, None, 1])
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.predict = self.inf(self.input_image)

        self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.ground_truth - self.predict)))
        self.gt_count = tf.reduce_sum(self.ground_truth)
        self.et_count = tf.reduce_sum(self.predict)
        self.MAE = tf.abs(self.gt_count - self.et_count)
        # Optimizer
        self.train_step = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)
        # Saver
        self.saver = tf.train.Saver(max_to_keep=5)
        self.best_saver = tf.train.Saver(max_to_keep=1)

    def data_pre_train(self, kind, dataset):
        img_path = (
            "./data/formatted_trainval/shanghaitech_part_"
            + dataset
            + "_patches_9/"
            + kind
            + "/"
        )
        den_path = (
            "./data/formatted_trainval/shanghaitech_part_"
            + dataset
            + "_patches_9/"
            + kind
            + "_den/"
        )
        print("loading", kind, "data from dataset", dataset, "...")
        img_names = os.listdir(img_path)
        img_num = len(img_names)

        data = []
        for i in range(1, img_num + 1):
            if i % 100 == 0:
                print(i, "/", img_num)
            name = img_names[i - 1]
            img = cv2.imread(img_path + name, 0)
            img = np.array(img)
            img = (img - 127.5) / 128
            den = np.loadtxt(open(den_path + name[:-4] + ".csv"), delimiter=",")
            den_quarter = np.zeros((int(den.shape[0] / 4), int(den.shape[1] / 4)))
            for i in range(len(den_quarter)):
                for j in range(len(den_quarter[0])):
                    for p in range(4):
                        for q in range(4):
                            den_quarter[i][j] += den[i * 4 + p][j * 4 + q]
            data.append([img, den_quarter])
        print("load", kind, "data from dataset", dataset, "finished")
        return data

    def data_pre_test(self, dataset):
        img_path = (
            "./data/original/shanghaitech/part_" + dataset + "_final/test_data/images/"
        )
        den_path = (
            "./data/original/shanghaitech/part_"
            + dataset
            + "_final/test_data/ground_truth_csv/"
        )
        print("loading test data from dataset", dataset, "...")
        img_names = os.listdir(img_path)
        img_num = len(img_names)

        data = []
        for i in range(1, img_num + 1):
            if i % 50 == 0:
                print(i, "/", img_num)
            name = "IMG_" + str(i) + ".jpg"
            img = cv2.imread(img_path + name, 0)
            img = np.array(img)
            img = (img - 127.5) / 128
            den = np.loadtxt(open(den_path + name[:-4] + ".csv"), delimiter=",")
            den_sum = np.sum(den)
            data.append([img, den_sum])

            # if i <= 2:
            # heatmap(den, i, dataset, 'act')

        print("load test data from dataset", dataset, "finished")
        return data

    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(
            x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )

    def inf(self, x):
        # s net ###########################################################
        w_conv1_1 = tf.get_variable("w_conv1_1", [5, 5, 1, 24])
        b_conv1_1 = tf.get_variable("b_conv1_1", [24])
        h_conv1_1 = tf.nn.relu(self.conv2d(x, w_conv1_1) + b_conv1_1)

        h_pool1_1 = self.max_pool_2x2(h_conv1_1)

        w_conv2_1 = tf.get_variable("w_conv2_1", [3, 3, 24, 48])
        b_conv2_1 = tf.get_variable("b_conv2_1", [48])
        h_conv2_1 = tf.nn.relu(self.conv2d(h_pool1_1, w_conv2_1) + b_conv2_1)

        h_pool2_1 = self.max_pool_2x2(h_conv2_1)

        w_conv3_1 = tf.get_variable("w_conv3_1", [3, 3, 48, 24])
        b_conv3_1 = tf.get_variable("b_conv3_1", [24])
        h_conv3_1 = tf.nn.relu(self.conv2d(h_pool2_1, w_conv3_1) + b_conv3_1)

        w_conv4_1 = tf.get_variable("w_conv4_1", [3, 3, 24, 12])
        b_conv4_1 = tf.get_variable("b_conv4_1", [12])
        h_conv4_1 = tf.nn.relu(self.conv2d(h_conv3_1, w_conv4_1) + b_conv4_1)

        # m net ###########################################################
        w_conv1_2 = tf.get_variable("w_conv1_2", [7, 7, 1, 20])
        b_conv1_2 = tf.get_variable("b_conv1_2", [20])
        h_conv1_2 = tf.nn.relu(self.conv2d(x, w_conv1_2) + b_conv1_2)

        h_pool1_2 = self.max_pool_2x2(h_conv1_2)

        w_conv2_2 = tf.get_variable("w_conv2_2", [5, 5, 20, 40])
        b_conv2_2 = tf.get_variable("b_conv2_2", [40])
        h_conv2_2 = tf.nn.relu(self.conv2d(h_pool1_2, w_conv2_2) + b_conv2_2)

        h_pool2_2 = self.max_pool_2x2(h_conv2_2)

        w_conv3_2 = tf.get_variable("w_conv3_2", [5, 5, 40, 20])
        b_conv3_2 = tf.get_variable("b_conv3_2", [20])
        h_conv3_2 = tf.nn.relu(self.conv2d(h_pool2_2, w_conv3_2) + b_conv3_2)

        w_conv4_2 = tf.get_variable("w_conv4_2", [5, 5, 20, 10])
        b_conv4_2 = tf.get_variable("b_conv4_2", [10])
        h_conv4_2 = tf.nn.relu(self.conv2d(h_conv3_2, w_conv4_2) + b_conv4_2)

        # l net ###########################################################
        w_conv1_3 = tf.get_variable("w_conv1_3", [9, 9, 1, 16])
        b_conv1_3 = tf.get_variable("b_conv1_3", [16])
        h_conv1_3 = tf.nn.relu(self.conv2d(x, w_conv1_3) + b_conv1_3)

        h_pool1_3 = self.max_pool_2x2(h_conv1_3)

        w_conv2_3 = tf.get_variable("w_conv2_3", [7, 7, 16, 32])
        b_conv2_3 = tf.get_variable("b_conv2_3", [32])
        h_conv2_3 = tf.nn.relu(self.conv2d(h_pool1_3, w_conv2_3) + b_conv2_3)

        h_pool2_3 = self.max_pool_2x2(h_conv2_3)

        w_conv3_3 = tf.get_variable("w_conv3_3", [7, 7, 32, 16])
        b_conv3_3 = tf.get_variable("b_conv3_3", [16])
        h_conv3_3 = tf.nn.relu(self.conv2d(h_pool2_3, w_conv3_3) + b_conv3_3)

        w_conv4_3 = tf.get_variable("w_conv4_3", [7, 7, 16, 8])
        b_conv4_3 = tf.get_variable("b_conv4_3", [8])
        h_conv4_3 = tf.nn.relu(self.conv2d(h_conv3_3, w_conv4_3) + b_conv4_3)

        # merge ###########################################################
        h_conv4_merge = tf.concat([h_conv4_1, h_conv4_2, h_conv4_3], 3)

        w_conv5 = tf.get_variable("w_conv5", [1, 1, 30, 1])
        b_conv5 = tf.get_variable("b_conv5", [1])
        h_conv5 = self.conv2d(h_conv4_merge, w_conv5) + b_conv5

        y_pre = h_conv5

        return y_pre

    def train(self, max_epoch):
        with tf.Session() as sess:
            if not os.path.exists("./model" + self.dataset):
                sess.run(tf.global_variables_initializer())
            else:
                saver = tf.train.Saver()
                saver.restore(sess, "model" + self.dataset + "/model.ckpt")

            data_train = self.data_pre_train("train", self.dataset)
            data_val = self.data_pre_train("val", self.dataset)

            best_mae = 10000
            for epoch in range(max_epoch):
                # training process
                epoch_mae = 0
                random.shuffle(data_train)
                start_time = time.time()
                for i in range(len(data_train)):
                    data = data_train[i]
                    x_in = np.reshape(
                        data[0], (1, data[0].shape[0], data[0].shape[1], 1)
                    )
                    y_ground = np.reshape(
                        data[1], (1, data[1].shape[0], data[1].shape[1], 1)
                    )

                    _, l, y_a, y_p, act_s, pre_s = sess.run(
                        [
                            self.train_step,
                            self.loss,
                            self.y_act,
                            self.y_pre,
                            self.act_sum,
                            self.pre_sum,
                            # self.MAE,
                        ],
                        feed_dict={self.x: x_in, self.y_act: y_ground},
                    )
                    if i % 500 == 0:
                        print("epoch", epoch, "step", i, "mae:", m)
                    epoch_mae += np.abs(np.subtract(y_a, y_p))
                epoch_mae /= len(data_train)
                d_time = round(time.time() - start_time)
                log_str = "Epoch[{}], time: {}, train_mae: {:.4f}".format(
                    epoch + 1, d_time, epoch_mae
                )
                print(log_str)
                log_file.write(log_str)
                log_file.flush()
                # validation process
                val_mae = 0
                val_mse = 0
                for i in range(len(data_val)):
                    data = data_val[i]
                    x_in = np.reshape(
                        data[0], (1, data[0].shape[0], data[0].shape[1], 1)
                    )
                    y_ground = np.reshape(
                        data[1], (1, data[1].shape[0], data[1].shape[1], 1)
                    )

                    act_s, pre_s, m = sess.run(
                        [self.act_sum, self.pre_sum, self.MAE],
                        feed_dict={self.x: x_in, self.y_act: y_ground},
                    )
                    val_mae += m
                    val_mse += (act_s - pre_s) * (act_s - pre_s)
                val_mae /= len(data_val)
                val_mse = math.sqrt(val_mse / len(data_val))
                print("epoch", epoch, "valid_mae:", val_mae, "valid_mse:", val_mse)
                
                if val_mae < best_mae:
                    best_mae = val_mae
                    print("best mae so far, saving model.")
                    saver = tf.train.Saver()
                    saver.save(sess, "model" + self.dataset + "/model.ckpt")
                else:
                    print("best mae:", best_mae)
                print("**************************")

    def test(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "model" + self.dataset + "/model.ckpt")
            data = self.data_pre_test(self.dataset)

            mae = 0
            mse = 0
            for i in range(1, len(data) + 1):
                if i % 20 == 0:
                    print(i, "/", len(data))
                d = data[i - 1]
                x_in = d[0]
                y_a = d[1]

                x_in = np.reshape(d[0], (1, d[0].shape[0], d[0].shape[1], 1))
                y_p_den = sess.run(self.y_pre, feed_dict={self.x: x_in})

                y_p = np.sum(y_p_den)

                # if i <= 2:
                # y_p_den = np.reshape(y_p_den, (y_p_den.shape[1], y_p_den.shape[2]))
                # heatmap(y_p_den, i, self.dataset, 'pre')
                mae += abs(y_a - y_p)
                mse += (y_a - y_p) * (y_a - y_p)

            mae /= len(data)
            mse = math.sqrt(mse / len(data))
            print("mae: ", mae)
            print("mse: ", mse)

    def get_val_data(self, img_list, gt_list):
        data = []
        for i in range(len(img_list)):
            img_path = img_list[i]
            gt_path = gt_list[i]
            img_data = self.ReadImage(img_path)
            gt_data = self.ScaleDmap(self.ReadDmap(gt_path))
            data.append([img_data, gt_data])
        return data

    def load_data_list(self, img_dir, dmap_dir):
        img_list = glob("{}/*.jpg".format(img_dir))
        img_list.sort()
        dmap_list = glob("{}/*.csv".format(dmap_dir))
        dmap_list.sort()

        def check_size(a, b):
            return len(a) != 0 and len(a) == len(b)

        if not check_size(img_list, dmap_list):
            raise ValueError(
                "Size error: img_list: {}, dmap_list: {}".format(
                    len(img_list), len(dmap_list)
                )
            )
        return img_list, dmap_list

    def run(self, sess, get_data, feed_data):
        """
        This function return get_data
        Ex: get_data = [model.optimizer, model.loss]
            _, loss = model.train(get_data, feed_data)
        """
        return sess.run(get_data, feed_data)

    def get_batch_patches(self, batch_size, input_shape, img_path, dmap_path):

        patch_height = int(round(input_shape[0]))
        patch_width = int(round(input_shape[1]))

        rand_img = self.ReadImage(img_path)
        rand_dmap = self.ReadDmap(dmap_path)

        if np.random.random() > 0.5:
            rand_img = np.fliplr(rand_img)
            rand_dmap = np.fliplr(rand_dmap)

        batch_img = np.zeros([batch_size, patch_height, patch_width, 1]).astype(
            "float32"
        )
        batch_dmap = np.zeros(
            [batch_size, int(patch_height / 4), int(patch_width / 4), 1]
        ).astype("float32")

        rand_img = rand_img.astype("float32")
        rand_dmap = rand_dmap.astype("float32")

        h, w = rand_img.shape

        for k in range(batch_size):
            # randomly select a box anchor
            w_rand = randint(0, w - patch_width)
            h_rand = randint(0, h - patch_height)

            pos = np.array([w_rand, h_rand])
            # crop
            img_norm = copy.deepcopy(
                rand_img[pos[1] : pos[1] + patch_height, pos[0] : pos[0] + patch_width]
            )
            dmap_temp = copy.deepcopy(
                rand_dmap[pos[1] : pos[1] + patch_height, pos[0] : pos[0] + patch_width]
            )

            # Scale this dmap first
            # print("dmap_temp: {}".format(dmap_temp.shape))
            if dmap_temp.shape[0] != patch_height or dmap_temp.shape[1] != patch_width:
                print("w_rand: {}, h_rand: {}, pos: {}\n".format(w_rand, h_rand, pos))
                print(rand_dmap.shape)
                print(dmap_temp.shape)
                print(dmap_path)
                print("{}, {}".format(patch_width, patch_height))
            dmap_temp = self.ScaleDmap(dmap_temp)
            batch_img[k, :, :, 0] = img_norm
            batch_dmap[k, :, :, 0] = dmap_temp
        return batch_img, batch_dmap

    def ScaleDmap(self, map_data):
        den_quarter = np.zeros((int(map_data.shape[0] / 4), int(map_data.shape[1] / 4)))
        for i in range(len(den_quarter)):
            for j in range(len(den_quarter[0])):
                for p in range(4):
                    for q in range(4):
                        den_quarter[i][j] += map_data[i * 4 + p][j * 4 + q]
        # print("Before: {}, After: {}".format(np.sum(map_data), np.sum(den_quarter)))
        return den_quarter

    def ReadImage(self, img_path):
        imArr = np.asarray(Image.open(img_path).convert("L"))
        return imArr

    def ReadDmap(self, map_path, scale=1):
        map_data = pd.read_csv(map_path, sep=",", header=None)
        map_data = np.asarray(map_data, dtype=np.float32)
        if scale != 1:
            map_data = self.ScaleDmap(map_data, scale=scale)
        return map_data

    def SaveImage(self, imgArr, save_path):
        img = Image.fromarray(np.array(imgArr * 255.0).astype("uint8"))
        img.save(save_path + ".jpg")

    def SaveMap(self, map_data, save_path):
        if len(map_data.shape) > 2:
            map_data = np.reshape(map_data, [map_data.shape[1], map_data.shape[2]])
        print("max: {}, min: {}".format(np.max(map_data), np.min(map_data)))
        d = np.max(map_data) - np.min(map_data)
        if d != 0:
            map_data = np.divide((map_data - np.min(map_data)), d)
        img = Image.fromarray(np.array(map_data * 255.0).astype("uint8"))
        img.save(save_path + ".jpg")

