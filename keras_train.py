import numpy as np
import sys
import os
import keras.backend as K
import datetime
import pickle
import time
import tensorflow.compat.v1 as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Reshape, Concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from random import shuffle
from tqdm import tqdm

from heatmap import *
from utils import *

scale = 1e3


def mae(y_true, y_pred):
    r = abs(K.sum(y_true) - K.sum(y_pred))
    return r / scale


def mse(y_true, y_pred):
    a = (K.sum(y_true) - K.sum(y_pred)) / scale
    return a * a


class KERAS_MCNN(object):
    def __init__(self, params):
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
        self._model = self._build()
        self.name = "Keras_MCNN"
        self.modeldir = os.path.join(self.checkpoint_dir, self.name)
        makedirs(self.modeldir)
        # build model

    def _build(self):
        inputs = Input(shape=(None, None, 1))
        conv_m = Conv2D(20, (7, 7), padding="same", activation="relu")(inputs)
        conv_m = MaxPooling2D(pool_size=(2, 2))(conv_m)
        conv_m = conv_m
        conv_m = Conv2D(40, (5, 5), padding="same", activation="relu")(conv_m)
        conv_m = MaxPooling2D(pool_size=(2, 2))(conv_m)
        conv_m = Conv2D(20, (5, 5), padding="same", activation="relu")(conv_m)
        conv_m = Conv2D(10, (5, 5), padding="same", activation="relu")(conv_m)
        # conv_m = Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(conv_m)

        conv_s = Conv2D(24, (5, 5), padding="same", activation="relu")(inputs)
        conv_s = MaxPooling2D(pool_size=(2, 2))(conv_s)
        conv_s = conv_s
        conv_s = Conv2D(48, (3, 3), padding="same", activation="relu")(conv_s)
        conv_s = MaxPooling2D(pool_size=(2, 2))(conv_s)
        conv_s = Conv2D(24, (3, 3), padding="same", activation="relu")(conv_s)
        conv_s = Conv2D(12, (3, 3), padding="same", activation="relu")(conv_s)
        # conv_s = Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(conv_s)

        conv_l = Conv2D(16, (9, 9), padding="same", activation="relu")(inputs)
        conv_l = MaxPooling2D(pool_size=(2, 2))(conv_l)
        conv_l = conv_l
        conv_l = Conv2D(32, (7, 7), padding="same", activation="relu")(conv_l)
        conv_l = MaxPooling2D(pool_size=(2, 2))(conv_l)
        conv_l = Conv2D(16, (7, 7), padding="same", activation="relu")(conv_l)
        conv_l = Conv2D(8, (7, 7), padding="same", activation="relu")(conv_l)
        # conv_l = Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(conv_l)

        conv_merge = Concatenate(axis=3)([conv_m, conv_s, conv_l])
        result = Conv2D(1, (1, 1), padding="same")(conv_merge)
        model = Model(inputs=inputs, outputs=result)

        adam = Adam(lr=1e-4)
        model.compile(loss="mse", optimizer=adam, metrics=[mae, mse])
        return model

    def train(self):
        model = self._model
        dev_test = self.dev_test
        # log file configuration
        log_fname = self.name + "keras_log.json"
        makedirs(self.log_dir)
        log_fname = os.path.join(self.log_dir, log_fname)
        log_file = open(log_fname, "w")
        # Configuration
        train_lines = generate_data(self.trainDataPath)
        val_lines = generate_data(self.valDataPath)
        val_data = pickle.load(open(val_lines[0], "rb"))
        start_time = time.time()
        x_train = []
        y_train = []
        x_val = []
        y_val = []
        print("Pre-loading data, this will take long ...")
        total_batch = len(train_lines)
        batch_size = 64
        val_num = len(val_data)
        if dev_test:
            total_batch = 1
            batch_size = 1
            val_num = 1

        def reshape_skip_batch(a):
            return np.reshape(a, [a.shape[0], a.shape[1], 1])

        for i in tqdm(range(val_num)):
            d = val_data[i]
            x = normalized_bit_wised(d[0])
            y = downsized_4(d[1]) * scale
            x_val.append(reshape_skip_batch(x))
            y_val.append(reshape_skip_batch(y))
        for i in tqdm(range(total_batch)):
            l = train_lines[i]
            train_data = pickle.load(open(l, "rb"))
            # shuffle(train_data)
            for d in train_data:
                # TODO
                x = normalized_bit_wised(d[0])
                y = downsized_4(d[1]) * scale
                x_train.append(reshape_skip_batch(x))
                y_train.append(reshape_skip_batch(y))

        x_val = np.array(x_val)
        y_val = np.array(y_val)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        d_time = round(time.time() - start_time)

        best_mae = 10000
        print("Start training")
        start_epoch = self._load_model(self.modeldir, model)
        period = 1
        if dev_test:
            self.end_epoch=start_epoch+1
            period = self.end_epoch + 100

        mc = ModelCheckpoint(
            os.path.join(self.modeldir, "weights.{epoch:02d}.hdf5"),
            save_weights_only=False,
            period=period,
        )

        training_history = model.fit(
            x_train,
            y_train,
            initial_epoch=start_epoch,
            epochs=self.end_epoch,
            batch_size=1,
            callbacks=[mc]
            # validation_data=(x_val, y_val),
        )

    def _writelog(self, log_str, log_file):
        print(log_str)
        log_file.write(log_str)
        log_file.flush()

    def test(self):
        model = self._model
        self._load_model(self.modeldir, model)
        val_lines = generate_data(self.valDataPath)
        val_data = pickle.load(open(val_lines[0], "rb"))
        # x_val = []
        # y_val = []
        print("Pre-loading data, this will take long ...")
        val_num = len(val_data)
        outdir = "output_keras"
        base_dir = os.path.join(outdir, "base")
        makedirs(outdir, base_dir)
        for i in range(val_num):
            d = val_data[i]
            x = normalized_bit_wised(d[0])
            # y = downsized_4(d[1])
            y_pre = model.predict(reshape_to_feed(x))
            y_pre /= scale
            gt_count = np.sum(d[1])
            et_count = np.sum(y_pre)
            save_name = "{}.png".format(i)
            base_path = os.path.join(base_dir, save_name)
            save_path = os.path.join(outdir, save_name)
            SaveImage(d[0], base_path)
            y_pre = upsized_4(reshape_to_eval(y_pre))
            check_consistent(d[0], y_pre, save_file=save_path, base=base_path)
            print("{}, GT: {}, ET: {}".format(i, gt_count, et_count))

    def _load_model(self, ckpt_dir, model, id=None):
        """
        If provide None, then the latest will be pick
        """
        files = os.listdir(ckpt_dir)
        if not files:
            return 0
        files.sort()
        ckpt_path = ""
        if id is not None:
            old_id = files[0].split(".")[1]
            ckpt_path = os.path.join(ckpt_dir, files[0].replace(old_id, str(id)))
        else:
            ckpt_path = os.path.join(ckpt_dir, files[-1])
            id = int(files[-1].split(".")[1])
        print("Load {}\n".format(ckpt_path))
        model.load_weights(ckpt_path)
        return id


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    ini_file = os.path.join(dirname, "param.ini")
    params = load_train_ini(ini_file)
    phase = params[0]["phase"]
    model = KERAS_MCNN(params[0])
    if phase == "test":
        model.test()
    elif phase == "train":
        model.train()
    else:
        raise ValueError("DOnt knOw phAse")
    # model.train()
