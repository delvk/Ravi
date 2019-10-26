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
from keras.callbacks import ModelCheckpoint, Callback, CSVLogger
from random import shuffle
from tqdm import tqdm

from heatmap import *
from utils import *
import copy

scale = 1e3


def reshape_skip_batch(a):
    return np.reshape(a, [a.shape[0], a.shape[1], 1])


def mae(y_true, y_pred):
    r = abs(K.sum(y_true) - K.sum(y_pred))
    return r / scale


def mse(y_true, y_pred):
    a = (K.sum(y_true) - K.sum(y_pred)) / scale
    return a * a


def create_patches_data(img_data, map_data, batch_x, batch_y):
    # Load
    width = 128
    height = 128
    # temp = [img_data, map_data]

    for k in range(16):
        if np.random.random() > 0.5:
            rand_img = np.fliplr(img_data)
            rand_map = np.fliplr(map_data)
        else:
            rand_img = img_data
            rand_map = map_data

        h, w = rand_img.shape
        w_rand = randint(0, w - width)
        h_rand = randint(0, h - height)
        pos = np.array([w_rand, h_rand])
        # crop
        img_norm = copy.deepcopy(
            rand_img[pos[1] : pos[1] + height, pos[0] : pos[0] + width]
        )
        map_temp = copy.deepcopy(
            rand_map[pos[1] : pos[1] + height, pos[0] : pos[0] + width]
        )
        map_temp = downsized_4(map_temp)
        x = normalized_bit_wised(img_norm)
        y = map_temp * scale
        batch_x.append(reshape_skip_batch(x))
        batch_y.append(reshape_skip_batch(y))


def generate_batches(file):
    """
    file: pkl files contain train_lines
    """
    lines = pickle.load(open(file, "rb"))
    shuffle(lines)
    # batch_y = []
    counter = 0
    while True:
        fname = lines[counter]
        # print("\n"+fname)
        counter = (counter + 1) % len(lines)
        data = pickle.load(open(fname, "rb"))
        #  data = [img_data, dmap]
        shuffle(data)
        for d in data:
            batch_x = []
            batch_y = []
            create_patches_data(d[0], d[1], batch_x, batch_y)
            yield np.array(batch_x), np.array(batch_y)


# model = Sequential()
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# train_files = [train_bundle_loc + "bundle_" + cb.__str__() for cb in range(nb_train_bundles)]
# gen = generate_batches(files=train_files, batch_size=batch_size)
# history = model.fit_generator(gen, samples_per_epoch=samples_per_epoch, nb_epoch=num_epoch,verbose=1, class_weight=class_weights)


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
        log_fname = self.name + "_training.log"
        makedirs(self.log_dir)
        log_fname = os.path.join(self.log_dir, log_fname)
        log_file = open(log_fname, "w")

        train_lines = generate_data(self.trainDataPath)
        val_lines = generate_data(self.valDataPath)

        x_train = []
        y_train = []
        x_val = []
        y_val = []

        print("Pre-loading data, this will take long ...")
        train_batch_num = len(train_lines)
        batch_size = 64
        val_batch_num = len(val_lines)
        if dev_test:
            train_batch_num = 1
            batch_size = 1
            val_batch_num = 1

        def reshape_skip_batch(a):
            return np.reshape(a, [a.shape[0], a.shape[1], 1])

        for i in tqdm(range(val_batch_num)):
            l = val_lines[i]
            val_data = pickle.load(open(l, "rb"))
            for d in val_data:
                x = normalized_bit_wised(d[0])
                y = downsized_4(d[1]) * scale
                x_val.append(reshape_skip_batch(x))
                y_val.append(reshape_skip_batch(y))

        for i in tqdm(range(train_batch_num)):
            l = train_lines[i]
            train_data = pickle.load(open(l, "rb"))
            for d in train_data:
                x = normalized_bit_wised(d[0])
                y = downsized_4(d[1]) * scale
                x_train.append(reshape_skip_batch(x))
                y_train.append(reshape_skip_batch(y))

        x_val = np.array(x_val, dtype=np.float64)
        y_val = np.array(y_val, dtype=np.float64)
        x_train = np.array(x_train, dtype=np.float64)
        y_train = np.array(y_train, dtype=np.float64)

        best_mae = 10000
        print("Start training")
        start_epoch = self._load_model(self.modeldir, model)
        period = 1
        if dev_test:
            self.end_epoch = start_epoch + 1
            period = self.end_epoch + 100
        mc = ModelCheckpoint(
            os.path.join(self.modeldir, "weights.{epoch:02d}.hdf5"),
            save_weights_only=False,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            period=period,
        )
        # history = LossHistory(log_file)
        csv_logger = CSVLogger('training.log')
        training_history = model.fit(
            x_train,
            y_train,
            initial_epoch=start_epoch,
            epochs=self.end_epoch,
            batch_size=1,
            callbacks=[mc, csv_logger],
            validation_data=(x_val, y_val),
            shuffle=True,
        )

    def _writelog(self, log_str, log_file):
        print(log_str)
        log_file.write(log_str)
        log_file.flush()

    def test(self, test_with_cropped=False, save_result=False):

        model = self._model
        self._load_model(self.modeldir, model)
        val_lines = generate_data(self.valDataPath)
        val_data = pickle.load(open(val_lines[0], "rb"))
        val_mae = 0.0
        # x_val = []
        # y_val = []
        print("Pre-loading data, this will take long ...")
        val_batch_num = len(val_data)
        outdir = "output_keras"
        base_dir = os.path.join(outdir, "base")
        makedirs(outdir, base_dir)
        d_time = 0
        for i in range(val_batch_num):
            d = val_data[i]
            if test_with_cropped:
                crop_x = crop_img(d[0])
                crop_y = crop_img(d[1])
                if len(crop_x) != len(crop_y):
                    raise ValueError("wrong size")
                sum_j = 0
                for j in range(len(crop_x)):
                    x_in = normalized_bit_wised(crop_x[j])

                    y_pre = model.predict(reshape_to_feed(x_in))

                    y_pre /= scale
                    gt_count = np.sum(crop_y[j])
                    et_count = np.sum(y_pre)
                    # save_name = "{}_{}.png".format(i,j)
                    # base_path = os.path.join(base_dir, save_name)
                    # save_path = os.path.join(outdir, save_name)
                    # SaveImage(crop_x[j], base_path)
                    # y_pre = upsized_4(reshape_to_eval(y_pre))
                    # check_consistent(crop_x[j], y_pre, save_file=save_path, base=base_path)
                    sum_j += et_count
                print("From cropped: {}, GT: {}, ET: {}".format(i, np.sum(d[1]), sum_j))

            x = normalized_bit_wised(d[0])
            start_time = time.time()
            y_pre = model.predict(reshape_to_feed(x))
            d_time += time.time() - start_time
            et_count = np.sum(y_pre) / scale

            if save_result:
                base_name = "{}.png".format(i)
                act_save_name = "{}_act.png".format(i)
                pre_save_name = "{}_pre.png".format(i)

                base_path = os.path.join(base_dir, base_name)
                act_save_path = os.path.join(outdir, act_save_name)
                pre_save_path = os.path.join(outdir, pre_save_name)
                SaveImage(d[0], base_path)
                y_pre = upsized_4(reshape_to_eval(y_pre))
                check_consistent(x, y_pre, save_file=pre_save_path, base=base_path)
                check_consistent(x, d[1], save_file=act_save_path, base=base_path)
            val_mae += np.abs(np.sum(d[1]) - et_count)
            print(
                "From NOT cropped: {}, GT: {}, ET: {}".format(i, np.sum(d[1]), et_count)
            )
            # if i == 0:
            #     break
        val_mae /= val_batch_num
        print("MAE: {}".format(val_mae))
        print(d_time)

    def test_from_dir(self, img_dir, gt_dir=None):
        self._load_model(self.modeldir, self._model)
        img_list = os.listdir(img_dir)
        mae = 0.0
        for name in img_list:
            img_path = os.path.join(img_dir, name)

            img_data = ReadImage(img_path, gray_scale=True)
            img_data = normalized_bit_wised(img_data)
            gt_data = None
            gt_path = None
            if gt_dir:
                temp = name.replace("jpg", "csv")
                gt_path = os.path.join(gt_dir, temp.replace("IMG", "DMAP"))
                gt_data = ReadMap(gt_path)

            predict = self._model.predict(reshape_to_feed(img_data))
            et_count = np.sum(predict)
            et_count /= scale

            if gt_dir:
                mae += np.abs(np.sum(gt_data) - et_count)
                print(
                    "{}, ground-truth: {}, predict: {}".format(
                        name, np.sum(gt_data), et_count
                    )
                )
            else:
                print("{}, predict: {}".format(name, et_count))

        if gt_dir:
            mean_mae = np.divide(mae, len(img_list))
            print("Mean mae: {}".format(mean_mae))

    def predict(self, img_path):
        x_in = ReadImage(img_path, gray_scale=True)
        x_in = reshape_to_feed(normalized_bit_wised(x_in))
        # start_time = time.time()
        y_pre = self._model.predict(x_in)
        # end_time = time.time() - start_time
        # print(end_time)

    def _load_model(self, ckpt_dir, model, id=None):
        """
        If provide None, then the latest will be pick
        """
        files = os.listdir(ckpt_dir)
        id_list = []
        if not files:
            return 0
        for f in files:
            id_list.append(int(f.split(".")[1]))
        id_list.sort()
        print(id_list)
        ckpt_path = ""
        old_id = files[0].split(".")[1]
        if id is not None:
            ckpt_path = os.path.join(ckpt_dir, files[0].replace(old_id, str(id)))
        else:
            ckpt_path = os.path.join(
                ckpt_dir, files[0].replace(old_id, str(id_list[-1]))
            )
        print("Load {}\n".format(ckpt_path))
        model.load_weights(ckpt_path)
        return id_list[-1]


class LossHistory(Callback):
    def __init__(self, log_file):
        self.log_file = log_file

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, epoch, logs={}):
        # self.losses.append(logs.get('loss'))
        # self.val_losses.append(logs.get('val_loss'))
        log_str = "{} - val_loss: {}\n".format(epoch, logs.get("val_loss"))
        self.log_file.write(log_str)


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    ini_file = os.path.join(dirname, "param.ini")
    params = load_train_ini(ini_file)
    phase = params[0]["phase"]
    model = KERAS_MCNN(params[0])
    if phase == "test":
        model.test_from_dir(params[0]["test_ImagePath"], params[0]["test_DmapPath"])
    elif phase == "train":
        model.train()
    elif phase == "predict":
        model.predict("raw/train_data/images/IMG_1.jpg")
    elif phase == "val":
        model.test()
    else:
        raise ValueError("DOnt knOw phAse")
    # model.train()
