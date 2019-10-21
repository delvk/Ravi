from __future__ import division
import numpy as np
import os
from tqdm import tqdm
from mcnn import MCNN
import tensorflow.compat.v1 as tf
from utils import *
import datetime
from heatmap import *
import pickle
from random import shuffle
import time


class MODEL(object):
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
        self.trainDataPath = params["trainDataPath"]
        self.valDataPath = params["valDataPath"]
        # private variables
        tf.reset_default_graph()
        self._model = MCNN(self.init_lr, self.init_lr)
        self._saver_path = os.path.join(self.checkpoint_dir, "continue")
        self._best_saver_path = os.path.join(self.checkpoint_dir, "best")
        self._name = self._model.name
        # build model

    def train(self):
        model = self._model
        dev_test = self.dev_test
        scale = 1e3
        # log file configuration
        log_fname = self.name + "_log.txt"
        makedirs(self.log_dir)
        log_fname = os.path.join(self.log_dir, log_fname)

        # Configuration
        init = tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer()
        )
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=0.65, allow_growth=True
        )
        with tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        ) as sess:
            sess.run(init)
            # Init value
            start_epoch = int(
                self._load_checkpoint(sess, model.saver, self._saver_path, self.name)
            )
            if start_epoch:
                log_file = open(log_fname, "a")
            elif os.path.exists(log_fname) and os.path.isfile(log_fname):
                if not dev_test:
                    raise ValueError(
                        "Log file already exist, make sure empty log dir to write new one"
                    )
                log_file = open(log_fname, "w")
                # pass
            else:
                log_file = open(log_fname, "w")
            log_file.write("{}\n".format(datetime.datetime.now()))

            best_mae = self._get_best_mae(self._best_saver_path)
            # for  v in tf.trainable_variables():
            #     print(v)
            # exit()
            #
            train_lines = generate_data(self.trainDataPath)
            print("Pre-loading validation data, this may take a while ...")
            val_lines = generate_data(self.valDataPath)
            val_data = pickle.load(open(val_lines[0], "rb"))
            makedirs("val_output")
            # dev_test
            if dev_test:
                self.end_epoch = start_epoch + 1
                train_lines = train_lines[:2]
                val_data = val_data[:1]

            print("Start epoch: {}/{}".format(start_epoch, self.end_epoch))
            print("Best result: {}".format(best_mae))
            print("Training total batches: {}".format(len(train_lines)))
            print("Validation total: {}".format(len(val_data)))
            # == training ==
            for epoch in range(start_epoch, self.end_epoch):
                print("------------------------------------")
                train_loss = 0.0
                train_mae = 0.0
                lr = calc_learning_rate(
                    init_lr=self.init_lr,
                    global_step=epoch,
                    decay_steps=self.decay_steps,
                    decay_rate=self.decay_rate,
                )
                get_data = [
                    model.train_step,
                    model.loss,
                    model.gt_count,
                    model.et_count,
                ]
                if np.mod(epoch, self.decay_steps) == 0 and epoch != 0:
                    self._load_checkpoint(
                        sess, model.best_saver, self._best_saver_path, self.name
                    )
                print("Epoch[{}]: ==> Started, Learning_rate: {}".format(epoch, lr))
                total_batch = len(train_lines)

                # shuffels batches
                shuffle(train_lines)
                num_train = 0
                start_time = time.time()
                for i in tqdm(range(total_batch)):
                    l = train_lines[i]
                    train_data = pickle.load(open(l, "rb"))
                    shuffle(train_data)
                    for data in train_data:
                        num_train += 1
                        x_in = reshape_to_feed(normalized_bit_wised(data[0]))
                        y_in = reshape_to_feed(downsized_4(data[1]) * scale)
                        feed_data = {
                            model.input_image: x_in,
                            model.ground_truth: y_in,
                            model.learning_rate: lr,
                        }
                        _, loss, gt_count, et_count = model.run(
                            sess, get_data, feed_data
                        )
                        train_loss += loss
                        train_mae += np.abs(np.subtract(gt_count, et_count / scale))
                train_loss = np.divide(train_loss, num_train)
                train_mae = np.divide(train_mae, num_train)
                d_time = round(time.time() - start_time)
                log_str = "Epoch[{}], time: {}, train_mae: {:.4f}\n".format(
                    epoch + 1, d_time, train_mae
                )
                self._writelog(log_str, log_file)

                if not dev_test:
                    self._save_checkpoint(
                        sess=sess,
                        saver=model.saver,
                        save_path=self._saver_path,
                        model_name=self.name,
                        step=epoch + 1,
                    )
                print("Validation processing ....")
                val_result = self.validation(sess, val_data, epoch, scale, model)
                log_str = "Epoch[{}]: Validation over {} samples -> mae: {}\n".format(
                    epoch, len(val_data), val_result
                )
                self._writelog(log_str, log_file)
                if val_result < best_mae:
                    print("Best result: {}".format(val_result))
                    best_mae = round(val_result, 2)
                    if not dev_test:
                        self._save_checkpoint(
                            sess=sess,
                            saver=model.best_saver,
                            save_path=self._best_saver_path,
                            model_name=self.name,
                            id=best_mae,
                        )
            log_file.close()

    def validation(self, sess, val_data, epoch, scale, model=MCNN):
        val_num = len(val_data)
        val_mae = 0.0
        rand_ints = np.random.randint(val_num, size=5)
        for i in range(val_num):
            data = val_data[i]
            x_in = reshape_to_feed(normalized_bit_wised(data[0]))
            y_gt = reshape_to_feed(downsized_4(data[1]))
            y_gt = np.reshape(data[1], (1, data[1].shape[0], data[1].shape[1], 1))
            get_data = [model.gt_count, model.et_count]
            feed_data = {model.input_image: x_in, model.ground_truth: y_gt}
            gt_count, et_count = model.run(sess, get_data, feed_data)
            if i in rand_ints:
                print("{}: GT: {}, ET: {}".format(i, gt_count, et_count / scale))
            mae = np.abs(np.subtract(gt_count, et_count / scale))
            val_mae += mae
        return np.divide(val_mae, val_num)

    def test(self):
        model = self._model
        test_mae = 0.0
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=0.65, allow_growth=True
        )
        with tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        ) as sess:
            # Load checkpoint
            ckpt = self._load_checkpoint(
                sess, model.best_saver, self._best_saver_path, self._name
            )
            if ckpt == 0:
                raise ValueError("Coudn't load checkpoint, check again")
            test_img_list, test_gt_list = model.load_data_list(
                self.test_ImagePath, self.test_DmapPath
            )
            test_num = len(test_img_list)
            output_dir = "test_output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for i in range(test_num):
                img_path = test_img_list[i]
                dmap_path = test_gt_list[i]
                img_name = img_path.split("/")[-1].split(".")[0]
                img_data = model.ReadImage(img_path)
                # normalized bit-wised
                img_data = normalized_bit_wised(img_data)
                # img_data = (img_data - 127.5) / 128
                img_feed = np.reshape(
                    img_data, [1, img_data.shape[0], img_data.shape[1], 1]
                )
                dmap_pre, et_count = model.run(
                    sess=sess,
                    get_data=[model.predict, model.et_count],
                    feed_data={model.input_image: img_feed},
                )
                gt_count = np.sum(model.ReadDmap(dmap_path))
                et_count = np.sum(dmap_pre)
                m = np.abs(gt_count - et_count)
                test_mae += m
                print("{}: GT: {:2f}, ET: {:.2f}".format(img_name, gt_count, et_count))
                # img_data = cv2.resize(img_data, (int(img_data.shape[1]/4), int(img_data.shape[0]/4)))
                dmap_pre = np.reshape(dmap_pre, [dmap_pre.shape[1], dmap_pre.shape[2]])
                dmap_pre = upsized_4(dmap_pre)
                save_name = (
                    img_name
                    + "_act_"
                    + str(int(gt_count))
                    + "_pre_"
                    + str(int(et_count))
                )
                save_name = os.path.join(output_dir, save_name)
                # check_consistent(img_data, dmap_pre, save_file=save_name, base=img_path)
            test_mae = np.divide(test_mae, len(test_img_list))
            print("MAE: {}".format(test_mae))

    def _save_checkpoint(self, sess, saver, save_path, model_name, step=None, id=None):
        """
        save checkpoint with step or id(best_mae)
        """
        print("Saving checkpoint, please wait ...")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if id == None:
            save_path = os.path.join(save_path, model_name)
            saver.save(sess, save_path, global_step=step)
            print(">>> Model saved: {}[{}]".format(model_name, step))
        else:
            save_path = os.path.join(save_path, "model-" + str(id))
            saver.save(sess, save_path)
            print(">>> Model saved: {}[{}]".format(model_name, id))

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

    def _writelog(self, log_str, log_file):
        print(log_str)
        log_file.write(log_str)
        log_file.flush()
