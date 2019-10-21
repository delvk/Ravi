import numpy
import tensorflow.compat.v1 as tf
import os
# import src
from model import MODEL
from my_mcnn import MCNN
from utils import calc_learning_rate, load_train_ini
# every things is relative to this file path
dirname = os.path.dirname(__file__)
def main(_):
    ini_file = os.path.join(dirname, 'param.ini')
    params = load_train_ini(ini_file)
    model = MODEL(params[0])
    model.train()
    # mcnn = MCNN(params[0])
    # mcnn.test()
    # param_set = load_train_ini(ini_file)[0]
    # print('====== Phase >>> {} <<< ======'.format(param_set['phase']))
    # checkpoint_dir = param_set['checkpoint_dir']
    # log_dir = param_set['log_dir']
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    
    # model = Model(param_set)
    # if param_set['phase'] == 'train':
    #     model.train()
    #     # model.predict("sample.jpg")
    # elif param_set['phase'] == 'test':
    #     model.test()
    #     # model.test(print_out=True,
    #     #            log_fname='test_cff_578.txt', save_path=None, max_people=999)
    #     # model.test_without_gt(print_out=True,
    #     #            save_path=result_dir, limit=999, max_people = 100)
    #     # img_path = "/home/jake/Desktop/temp/jj.jpg"
        # model.predict(img_path, save_path=result_dir, file_name="khuongdeptrai")

if __name__ == '__main__':
    tf.app.run()    