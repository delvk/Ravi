import configparser
import os
import numpy as np
from heatmap import *
from PIL import Image
from pyheatmap.heatmap import HeatMap
import pandas as pd
from random import *
import pickle
import copy
import tensorflow.compat.v1 as tf


def makedirs(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def load_train_ini(path):
    config = configparser.ConfigParser()
    config.read(path)
    param_sections = []
    sections = config.sections()
    # print(sections[0])
    for i in range(len(sections)):
        s_name = sections[i]
        s_dict = dict(
            phase=config.get(s_name, "phase"),
            dev_test=config.getboolean(s_name, "dev_test"),
            batch_size=config.getint(s_name, "batch_size"),
            patch_size=config.getint(s_name, "patch_size"),
            patch_width=config.getint(s_name, "patch_width"),
            patch_height=config.getint(s_name, "patch_height"),
            train_ImagePath=config.get(s_name, "train_ImagePath"),
            train_DmapPath=config.get(s_name, "train_DmapPath"),
            trainDataPath=config.get(s_name, "trainDataPath"),
            val_ImagePath=config.get(s_name, "val_ImagePath"),
            valDataPath=config.get(s_name, "valDataPath"),
            val_DmapPath=config.get(s_name, "val_DmapPath"),
            test_ImagePath=config.get(s_name, "test_ImagePath"),
            test_DmapPath=config.get(s_name, "test_DmapPath"),
            lr=config.getfloat(s_name, "learning_rate"),
            decay_steps=config.getint(s_name, "decay_steps"),
            decay_rate=config.getfloat(s_name, "decay_rate"),
            end_epoch=config.getint(s_name, "end_epoch"),
            log_dir=config.get(s_name, "log_dir"),
            checkpoint_dir=config.get(s_name, "checkpoint_dir"),
            model_name=config.get(s_name, "model_name"),
        )
        param_sections.append(s_dict)
    return param_sections


def test_load_ini():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, os.pardir, "tr_param.ini")
    a = load_train_ini(path)
    print(a)


def calc_learning_rate(init_lr, global_step, decay_steps, decay_rate):
    lr = init_lr
    for i in range(int(np.divide(global_step, decay_steps))):
        lr *= decay_rate
    return lr


def check_consistent(img, map, save_file="check.png", base=None):
    if type(img) is str:
        img = ReadImage(img, gray_scale=True)
    if type(map) is str:
        map = ReadMap(map)
    if not save_file.endswith(".png"):
        save_file += ".png"
    heatmap(img, map, save_file, "act", base=base)


def ReadImage(img_path, gray_scale):
    img_data = Image.open(img_path)
    if gray_scale:
        img_data = img_data.convert("L")
    else:
        img_data = img_data.convert("RGB")
    return np.asarray(img_data, dtype=np.float32)


def ReadMap(map_path):
    map_data = pd.read_csv(map_path, sep=",", header=None)
    map_data = np.asarray(map_data, dtype=np.float32)
    return map_data


def SaveImage(imgArr, save_path, mode="RGB"):
    if not save_path.endswith(".png"):
        save_path += ".png"
    img = Image.fromarray(np.array(imgArr).astype("uint8"))
    img = img.convert(mode)
    img.save(save_path)


def SaveMap(map_data, save_path):
    if not save_path.endswith(".csv"):
        save_path += ".csv"
    if len(map_data.shape) > 2:
        map_data = np.reshape(map_data, [map_data.shape[1], map_data.shape[2]])
    print("max: {}, min: {}".format(np.max(map_data), np.min(map_data)))
    d = np.max(map_data) - np.min(map_data)
    if d != 0:
        map_data = np.divide((map_data - np.min(map_data)), d)
    img = Image.fromarray(np.array(map_data * 255.0).astype("uint8"))
    img.save(save_path + ".jpg")


def heatmap(img, map, save_path, info, base=None):
    assert base is None or os.path.isfile(base)
    if not save_path.endswith(".png"):
        save_path += ".png"
    if info == "pre":
        den_resized = np.zeros((map.shape[0] * 4, map.shape[1] * 4))
        for i in range(den_resized.shape[0]):
            for j in range(den_resized.shape[1]):
                den_resized[i][j] = map[int(i / 4)][int(j / 4)] / 16
        map = den_resized
    # map = np.random.randint(2, size=(100, 100))
    count = np.sum(map)
    if np.max(map) != 0.0:
        map = np.divide(map * 10, np.max(map))
    w = img.shape[1]
    h = img.shape[0]

    data = []
    for j in range(len(map)):
        for i in range(len(map[0])):
            for k in range(int(map[j][i])):
                data.append([i + 1, j + 1])
    hm = HeatMap(data, base)
    hm.heatmap(save_as=save_path)


# def ReadPickle(path):
#     f = open(path, "r")
#     lines = f.readlines()
#     return lines


def generate_data(save_path, batch_size=None, img_dir=None, gt_dir=None, replace=None):
    """
    Check if file in save_path exist, if Yes => Return it
    If not, create and save 
    Return: list([img_data, gt_data])
    """
    if os.path.exists(save_path) and os.path.isfile(save_path):
        return pickle.load(open(save_path, "rb"))

    assert os.path.isdir(img_dir) and os.path.isdir(gt_dir)
    assert batch_size is not None
    makedirs(os.path.dirname(save_path))
    paths = []
    data = []
    img_names = os.listdir(img_dir)
    # shuffle(img_names)
    if np.mod(len(img_names), batch_size) != 0:
        print(np.mod(len(img_names), batch_size))
        raise ValueError(
            "Please provide batch size mod = 0 to total imgs {}".format(len(img_names))
        )
    for i in range(0, len(img_names), batch_size):
        names = img_names[i : i + batch_size]
        data = []
        for j in range(len(names)):
            img_path = os.path.join(img_dir, names[j])
            den_path = os.path.join(gt_dir, names[j][:-4] + ".csv")
            if replace:
                den_path = den_path.replace("IMG", replace)
            img_data = ReadImage(img_path, gray_scale=True)
            den = ReadMap(den_path)
            # count = np.sum(den)
            data.append([img_data, den])
        count_temp = len(data)
        save_name = save_path[:-4] + "_" + str(batch_size) + "_" + str(i) + ".pkl"
        paths.append(save_name)
        with open(save_name, "wb") as ff:
            pickle.dump(data, ff)
        # test
        count_test = len(pickle.load(open(save_name, "rb")))
        if count_temp == count_test:
            print("Save success {} data to {}".format(count_temp, save_name))
        else:
            print("Something went wrong at {}".format(save_name))
    with open(save_path, "wb") as f:
        pickle.dump(paths, f)
    return pickle.load(open(save_path, "rb"))


def downsized_4(den):
    den_quarter = np.zeros((int(den.shape[0] / 4), int(den.shape[1] / 4)))
    for i in range(len(den_quarter)):
        for j in range(len(den_quarter[0])):
            for p in range(4):
                for q in range(4):
                    den_quarter[i][j] += den[i * 4 + p][j * 4 + q]
    return den_quarter


def upsized_4(den):
    den_resized = np.zeros((den.shape[0] * 4, den.shape[1] * 4))
    for i in range(den_resized.shape[0]):
        for j in range(den_resized.shape[1]):
            den_resized[i][j] = den[int(i / 4)][int(j / 4)] / 16
    return den_resized


def normalized_bit_wised(data, v_min=-1, v_max=+1):
    return np.interp(data, (data.min(), data.max()), (v_min, v_max))


def load_and_check(pkl_file, id, total=1):
    data = pickle.load(open(pkl_file, "rb"))
    dirname = os.path.dirname(pkl_file)
    basedir = os.path.join(dirname, "base")
    checkdir = os.path.join(dirname, "check")
    makedirs(basedir, checkdir)
    shuffle(data)
    for i in range(min(total, len(data))):
        d = data[i]
        base_path = os.path.join(basedir, "IMG_" + str(id) + "_" + str(i) + ".png")
        save_path = os.path.join(checkdir, str(id) + "_" + str(i) + ".png")
        save_img = SaveImage(d[0], base_path)
        check_consistent(base_path, d[1], save_file=save_path, base=base_path)


def reshape_to_feed(tensor):
    return np.reshape(tensor, [1, tensor.shape[0], tensor.shape[1], 1])


def reshape_to_eval(tensor):
    return np.reshape(tensor, [tensor.shape[1], tensor.shape[2]])


def create_data(
    src_img_dir,
    src_gt_dir,
    des_img_dir,
    des_gt_dir,
    patch_size,
    patch_shape,
    replace=None,
):
    makedirs(des_img_dir, des_gt_dir)
    if os.listdir(des_img_dir) or os.listdir(des_gt_dir):
        return
        # raise ValueError("Directory is not empty ")

    height = patch_shape[0]
    width = patch_shape[1]
    img_names = os.listdir(src_img_dir)

    for j in range(len(img_names)):
        name = img_names[j][:-4]
        img_path = os.path.join(src_img_dir, name + ".jpg")
        den_path = os.path.join(src_gt_dir, name + ".csv")
        if replace:
            den_path = den_path.replace("IMG", replace)
        rand_img = ReadImage(img_path, gray_scale=False)
        rand_map = ReadMap(den_path)
        for k in range(patch_size):
            if np.random.random() > 0.5:
                rand_img = np.fliplr(rand_img)
                rand_map = np.fliplr(rand_map)
            h, w, c = rand_img.shape

            w_rand = randint(0, w - width)
            h_rand = randint(0, h - height)

            pos = np.array([w_rand, h_rand])
            # crop
            img_norm = copy.deepcopy(
                rand_img[pos[1] : pos[1] + height, pos[0] : pos[0] + width, :]
            )
            map_temp = copy.deepcopy(
                rand_map[pos[1] : pos[1] + height, pos[0] : pos[0] + width]
            )
            id = str(j) + "_" + str(k)
            SaveImage(img_norm, save_path=os.path.join(des_img_dir, id + ".png"))
            df = pd.DataFrame(map_temp)
            df.to_csv(os.path.join(des_gt_dir, id + ".csv"), index=None, header=None)


def prep_data_for_keras_train(phase):
    src_img_dir = "./raw/" + phase + "_data/images"
    src_gt_dir = "./raw/" + phase + "_data/dmap"
    des_img_dir = "./cooked/" + phase + "_data/images"
    des_gt_dir = "./cooked/" + phase + "_data/dmap"

    batch_size = 100
    patch_size = 16
    patch_shape = [128, 128]
    replace = None

    if phase == "val":
        batch_size = 116
        replace = "DMAP"
    create_data(
        src_img_dir,
        src_gt_dir,
        des_img_dir,
        des_gt_dir,
        patch_size,
        patch_shape,
        replace="DMAP",
    )
    pickle_link = os.path.join("pickle", phase, phase + ".pkl")
    lines = generate_data(
        pickle_link,
        batch_size=batch_size,
        img_dir=des_img_dir,
        gt_dir=des_gt_dir,
        replace="DMAP",
    )
    for i in range(len(lines)):
        l = lines[i]
        print("handle {}".format(l))
        load_and_check(l, i, 2)


def crop_img_from_path(img_path, chopsize=128):
    img = Image.open(img_path)
    width, height = img.size
    name = img_path.split("/")[-1]
    # Save Chops of original image
    for x0 in range(0, width, chopsize):
        for y0 in range(0, height, chopsize):
            box = (
                x0,
                y0,
                x0 + chopsize if x0 + chopsize < width else width - 1,
                y0 + chopsize if y0 + chopsize < height else height - 1,
            )
            print("%s %s" % (name, box))
            img.crop(box).save(
                "zchop.%s.x%03d.y%03d.jpg" % (name.replace(".jpg", ""), x0, y0)
            )


def crop_img(img_data, chopsize=128):
    data = []
    h, w = img_data.shape
    for x in range(0, w, chopsize):
        for y in range(0, h, chopsize):
            x1= x + chopsize if x + chopsize < w else w - 1
            y1 = y + chopsize if y + chopsize < h else h - 1
            cropped_data = copy.deepcopy(
                img_data[y : y1, x : x1]
            )
            data.append(cropped_data)
    return data
def crop_img_only(img_path, save_path):
    j =1
    rand_img = ReadImage(img_path, gray_scale=False)
    for k in range(16):
        if np.random.random() > 0.5:
            rand_img = np.fliplr(rand_img)
            # rand_map = np.fliplr(rand_map)
        h, w, c = rand_img.shape

        w_rand = randint(0, w - 128)
        h_rand = randint(0, h - 128)

        pos = np.array([w_rand, h_rand])
        # crop
        img_norm = copy.deepcopy(
            rand_img[pos[1] : pos[1] + 128, pos[0] : pos[0] + 128, :]
        )
        id = str(j) + "_" + str(k)
        SaveImage(img_norm, save_path=os.path.join(save_path,id+".png"))
        
if __name__ == "__main__":
    img_path = 'raw/train_data/images/IMG_1.jpg'
    save_path ='temp'
    crop_img_only(img_path, save_path)
    # crop_img(img_path)

    # variables_in_checkpoint = tf.train.list_variables('/home/jake/Desktop/Projects/Latex/LuanVan/demo/WORKINGON/checkpoint/continue')
    # for v in variables_in_checkpoint:
    #     print(v)
    # x = tf.constant(5.0, shape=[])
    # w = tf.constant([15.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    # xw = tf.divide(w, x)
    # # max_in_rows = tf.reduce_max(xw, 1)

    # sess = tf.Session()
    # print(sess.run(xw))
    # ==> [[0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
    #      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
    #      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
    #      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
    #      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]]

    # print sess.run(max_in_rows)
    # ==> [25.0, 25.0, 25.0, 25.0, 25.0]
