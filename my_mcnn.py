import numpy as np
import cv2
import tensorflow.compat.v1 as tf
import os
import random
import math
import sys
from heatmap import *
import os
from tqdm import tqdm
from utils import calc_learning_rate


class MCNN:
    def __init__(self, params):
        self.name = params["model_name"]
        self.checkpoint_dir = params["checkpoint_dir"]
        self.log_dir = params["log_dir"]
        self.batch_size = params["batch_size"]
        self.patch_size = params["patch_size"]
        self.patch_shape = [params["patch_height"], params["patch_width"]]
        self.train_ImagePath = params["train_ImagePath"]
        self.train_DmapPath = params["train_DmapPath"]
        self.val_ImagePath = params["val_ImagePath"]
        self.val_DmapPath = params["val_DmapPath"]
        self.test_ImagePath = params["test_ImagePath"]
        self.test_DmapPath = params["test_DmapPath"]
        self.decay_steps = params["decay_steps"]
        self.decay_rate = params["decay_rate"]
        self.end_epoch = params["end_epoch"]
        self.dev_test = params["dev_test"]
        self.init_lr = params["lr"]

        # private variables
        self._saver_path = os.path.join(self.checkpoint_dir, "continue")
        self._best_saver_path = os.path.join(self.checkpoint_dir, "best")

        # private operations
        self._makedirs(self._saver_path, self._best_saver_path)
        self._build()

    def _build(self):
        self.x = tf.placeholder(tf.float32, [None, None, None, 1])
        self.y_act = tf.placeholder(tf.float32, [None, None, None, 1])
        self.y_pre = self.inf(self.x)
        self.lr = tf.placeholder(tf.float32, [])
        self.loss = tf.losses.mean_squared_error(self.y_act, self.y_pre)
        self.act_sum = tf.reduce_sum(self.y_act)
        self.pre_sum = tf.reduce_sum(self.y_pre)
        self.MAE = tf.abs(self.act_sum - self.pre_sum)

    def _makedirs(self, *paths):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    def _data_pre_load(self, img_dir, gt_dir):
        img_names = os.listdir(img_dir)
        data = []
        for name in img_names:
            img_data = np.asarray(cv2.imread(name, 0), dtype=np.float32)
            den_path = os.path.join(gt_dir, name[:-4] + ".csv")
            den = np.loadtxt(open(den_path), delimiter=",")
            den_quarter = np.zeros((int(den.shape[0] / 4), int(den.shape[1] / 4)))
            for i in range(len(den_quarter)):
                for j in range(len(den_quarter[0])):
                    for p in range(4):
                        for q in range(4):
                            den_quarter[i][j] += den[i * 4 + p][j * 4 + q]
            data.append([img, den_quarter])
        print("load finished")
        return data

    def data_pre_train(self, kind, dataset):
        img_path = "cooked_data/" + kind + "/images/"
        den_path = "cooked_data/" + kind + "/ground_truth/"
        print("loading " + kind + " data ...")
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
            if kind == "val":
                name = name.replace("IMG", "DMAP")
            den = np.loadtxt(open(den_path + name[:-4] + ".csv"), delimiter=",")
            den_quarter = np.zeros((int(den.shape[0] / 4), int(den.shape[1] / 4)))
            for i in range(len(den_quarter)):
                for j in range(len(den_quarter[0])):
                    for p in range(4):
                        for q in range(4):
                            den_quarter[i][j] += den[i * 4 + p][j * 4 + q]
            data.append([img, den_quarter])
        print("load finished")
        return data

    def data_pre_test(self, dataset):
        img_path = "raw/test_data/images/"
        den_path = "raw/test_data/dmap/"
        print("loading test data ...")
        img_names = os.listdir(img_path)
        img_num = len(img_names)

        data = []
        for i in range(1, img_num + 1):
            if i % 50 == 0:
                print(i, "/", img_num)
            name = "IMG_" + str(i) + ".jpg"
            img = cv2.imread(img_path + name, 0)
            name = name.replace("IMG", "DMAP")
            img = np.array(img)
            img = (img - 127.5) / 128
            den = np.loadtxt(open(den_path + name[:-4] + ".csv"), delimiter=",")
            den_sum = np.sum(den)
            data.append([img, den_sum])

            # if i <= 2:
            # heatmap(den, i, dataset, 'act')

        print("load test data finished")
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

    def _load_checkpoint(self, sess, saver, save_path, model_name):
        """
        Return last checkpoint index or 0 if none
        """
        print(" [*] Reading checkpoint...")

        last_cp = tf.train.latest_checkpoint(save_path)
        if last_cp:
            saver.restore(sess, last_cp)
            last_cp_idx = last_cp[last_cp.rfind("-") + 1 :]
            print("Restore {}[{}]".format(model_name, last_cp_idx))
            return last_cp_idx
        else:
            print("Failed")
            return 0

    def _save_checkpoint(self, sess, saver, save_path, model_name, step=None, id=None):
        """
        save checkpoint with step or id(best_mae)
        """
        print("Saving checkpoint, please wait ...")
        self._makedirs(save_path)
        if id == None:
            save_path = os.path.join(save_path, model_name)
            saver.save(sess, save_path, global_step=step)
            print(">>> Model saved: {}[{}]".format(model_name, step))
        else:
            save_path = os.path.join(save_path, "model-" + str(id))
            saver.save(sess, save_path)
            print(">>> Model saved: {}[{}]".format(model_name, id))

    def _get_best_mae(self, path):
        """
        - best result dir: dir to the best model checkpoint
        return 999 if couldn't find one 
        """
        best_mae = 999
        if not os.path.exists(path) and not os.path.isdir(path):
            os.makedirs(path)
        else:
            last_cp = tf.train.latest_checkpoint(path)
            if last_cp:
                best_mae = float(last_cp[last_cp.rfind("-") + 1 :])
        return best_mae

    def train(self):
        train_step = tf.train.AdamOptimizer(self.init_lr).minimize(self.loss)
        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=1000)
            best_saver = tf.train.Saver(max_to_keep=5)
            start_epoch = int(
                self._load_checkpoint(sess, saver, self._saver_path, self.name)
            )
            if not start_epoch:
                sess.run(tf.global_variables_initializer())

            if self.dev_test:
                data_train = self.data_pre_train("train_10", "B")
                data_val = self.data_pre_train("val", "B")
                self.end_epoch = start_epoch + 1
            else:
                data_train = self.data_pre_train("train", "B")
                data_val = self.data_pre_train("val", "B")

            best_mae = self._get_best_mae(self._best_saver_path)
            rand_idx = np.arange(len(data_train))
            num_train = len(rand_idx)
            for epoch in range(start_epoch, self.end_epoch + 1):
                # training process
                epoch_mae = 0.0
                lr = calc_learning_rate(
                    self.init_lr, epoch, self.decay_steps, self.decay_rate
                )

                np.random.shuffle(rand_idx)
                print("Epoch[{}]. => Started !".format(epoch))
                for i in tqdm(rand_idx):
                    data = data_train[i]
                    x_in = np.reshape(
                        data[0], (1, data[0].shape[0], data[0].shape[1], 1)
                    )
                    y_ground = np.reshape(
                        data[1], (1, data[1].shape[0], data[1].shape[1], 1)
                    )

                    _, loss, y_a, y_p, act_s, pre_s, mae = sess.run(
                        [
                            train_step,
                            self.loss,
                            self.y_act,
                            self.y_pre,
                            self.act_sum,
                            self.pre_sum,
                            self.MAE,
                        ],
                        feed_dict={self.x: x_in, self.y_act: y_ground},
                    )
                    epoch_mae += mae
                epoch_mae = np.divide(epoch_mae, num_train)
                print("Epoch", epoch + 1, "train_mae:", epoch_mae)
                # if np.sum(epoch, 2) == 0:
                self._save_checkpoint(
                    sess, saver, self._saver_path, self.name, step=epoch + 1, id=None
                )
                # Validation
                val_mae = self._val(sess, data_val, step=20)
                print("Epoch[{}], valid_mae: {}".format(epoch, val_mae))
                if val_mae < best_mae:
                    best_mae = round(val_mae, 2)
                    print("Best mae so far, saving model.")
                    self._save_checkpoint(
                        sess=sess,
                        saver=best_saver,
                        save_path=self._best_saver_path,
                        model_name=self.name,
                        step=None,
                        id=best_mae,
                    )
                else:
                    print("Best mae:", best_mae)
                print("**************************")

    def _val(self, sess, data_val, step=20):
        # validation process
        val_mae = 0.0
        val_mse = 0.0
        for i in range(len(data_val)):
            data = data_val[i]
            x_in = np.reshape(data[0], (1, data[0].shape[0], data[0].shape[1], 1))
            y_ground = np.reshape(data[1], (1, data[1].shape[0], data[1].shape[1], 1))

            act_s, pre_s, m = sess.run(
                [self.act_sum, self.pre_sum, self.MAE],
                feed_dict={self.x: x_in, self.y_act: y_ground},
            )
            if np.mod(i, step) == 0:
                print("{}, GT: {}, ET: {}".format(i, act_s, pre_s))
            val_mae += m
            val_mse += (act_s - pre_s) * (act_s - pre_s)
        val_mae /= len(data_val)
        # val_mse = math.sqrt(val_mse / len(data_val))
        return val_mae

    def test(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()

            saver.restore(sess, self._best_saver_path+ "/model.ckpt")
            data = self.data_pre_test("B")

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

                y_p_den = np.reshape(y_p_den, (y_p_den.shape[1], y_p_den.shape[2]))
                heatmap(y_p_den, i, "B", "pre")
                print("{}, GT: {}, ET: {}".format(i, y_a, y_p))
                mae += abs(y_a - y_p)
                mse += (y_a - y_p) * (y_a - y_p)

            mae /= len(data)
            mse = math.sqrt(mse / len(data))
            print("mae: ", mae)
            print("mse: ", mse)

