import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras_anomaly_detection.library.plot_utils import visualize_reconstruction_error
from keras_anomaly_detection.library.recurrent import CnnLstmAutoEncoder
from keras_anomaly_detection.library.recurrent import LstmAutoEncoder
from keras_anomaly_detection.library.convolutional import Conv1DAutoEncoder
from keras_anomaly_detection.library.recurrent import BidirectionalLstmAutoEncoder
from keras_anomaly_detection.library.feedforward import FeedForwardAutoEncoder
import demo.RM.information as info

import matplotlib.pyplot as plt
import demo.RM.data as data
import time
DO_TRAINING = False
RANDOM_SEED = 42
WINDOW_SIZE = 5

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path

def AutoEncoder_test(X_data, Y_data, sub_output_dir, num, model_name, ae) :

    model_dir_path = sub_output_dir + model_name

    anomaly_dir = model_dir_path + 'anomaly/'
    png_dir  = model_dir_path + 'png/'

    create_directory(model_dir_path)
    create_directory(anomaly_dir)
    create_directory(png_dir)

    # fit the data and save model into model_dir_path
    ae.fit(X_data, model_dir_path=model_dir_path, estimated_negative_sample_ratio=0.9)

    # load back the model saved in model_dir_path detect anomaly
    #ae.load_model(model_dir_path)
    anomaly_information = ae.anomaly(X_data)
    reconstruction_error = []
    file_name_info = anomaly_dir + str(num) + '_anomaly.txt'
    f = open(file_name_info, mode='at')
    f0 = open(model_dir_path + '0_anomaly.txt', mode='at')
    f1 = open(model_dir_path + '1_anomaly.txt', mode='at')
    f2 = open(model_dir_path + '2_anomaly.txt', mode='at')
    f3 = open(model_dir_path + '3_anomaly.txt', mode='at')
    f4 = open(model_dir_path + '4_anomaly.txt', mode='at')
    f5 = open(model_dir_path + '5_anomaly.txt', mode='at')
    f6 = open(model_dir_path + '6_anomaly.txt', mode='at')
    f7 = open(model_dir_path + '7_anomaly.txt', mode='at')

    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        temp_str = '# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')'
        #print(temp_str)
        f.write(temp_str + '\n')
        index = Y_data[idx]
        anomal_str = str(index) + ',' + str(dist)
        if index == 0 :
            f0.write(anomal_str+'\n')
        if index == 1 :
            f1.write(anomal_str+'\n')
        if index == 2 :
            f2.write(anomal_str+'\n')
        if index == 3 :
            f3.write(anomal_str+'\n')
        if index == 4 :
            f4.write(anomal_str+'\n')
        if index == 5 :
            f5.write(anomal_str+'\n')
        if index == 6 :
            f6.write(anomal_str+'\n')
        if index == 7 :
            f7.write(anomal_str+'\n')    
        reconstruction_error.append(dist)
    f.close()
    f0.close()
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()
    f7.close()
    png_name_info = png_dir + str(num) + '_anomaly.png'
    visualize_reconstruction_error(reconstruction_error, ae.threshold, Y_data, png_name_info, WINDOW_SIZE)


def main_test(db_file_name, sub_output_dir, COLUM_LIST, ERROR_LIST):
    np.random.seed(RANDOM_SEED)

    if os.path.exists(sub_output_dir):
        print(sub_output_dir + 'Already exist')
    else:
        create_directory(sub_output_dir)

    info.save_infomation(sub_output_dir, date_str, COLUM_LIST, ERROR_LIST)
    data.create_data(db_file_name, sub_output_dir, COLUM_LIST, ERROR_LIST)
    #data_dir_path = './data'
    #model_dir_path = './models'
    datasets_dict = data.load_data(sub_output_dir)
    X_data = datasets_dict['anomaly'][0]
    Y_data = datasets_dict['anomaly'][1]

    X_len = len(X_data)
    for i in range (X_len) :
        x_ = X_data[0]
        y_ = Y_data[0]
        scaler = MinMaxScaler()
        X_scaler_data = scaler.fit_transform(x_)

        model_name = 'LstmAutoEncoder/'
        ae = LstmAutoEncoder()
        AutoEncoder_test(X_scaler_data, y_, sub_output_dir, i, model_name, ae)

        model_name = 'CnnLstmAutoEncoder/'
        ae = CnnLstmAutoEncoder()
        AutoEncoder_test(X_scaler_data, y_, sub_output_dir, i, model_name, ae)

        model_name = 'Conv1DAutoEncoder/'
        ae = Conv1DAutoEncoder()
        AutoEncoder_test(X_scaler_data, y_, sub_output_dir, i, model_name, ae)

        model_name = 'BidirectionalLstmAutoEncoder/'
        ae = BidirectionalLstmAutoEncoder()
        AutoEncoder_test(X_scaler_data, y_, sub_output_dir, i, model_name, ae)

        model_name = 'FeedForwardAutoEncoder/'
        ae = FeedForwardAutoEncoder()
        AutoEncoder_test(X_scaler_data, y_, sub_output_dir, i, model_name, ae)
    info.save_information_done(sub_output_dir)



#if __name__ == '__main__':
#    main()
DATA_DIR = './data/'
RESULT_DIR = './results/'
date_str = time.strftime("%m%d_%H%M")
#test_str = DATASET_NAME[TEST_NUM]+ "/" + date_str
output_dir = RESULT_DIR + date_str + '/'
db_file_name = './data/ub.csv'
output_info_dir = output_dir + 'info.txt'

if os.path.exists(output_dir):
    print(output_dir + 'Already done')
else:
    data.create_directory(output_dir)

dataset_name = 'make_data'
date_str = time.strftime("%m%d_%H%M")
sub_output_dir = output_dir + date_str + '/'
COLUM_LIST = ["courseoption1","courseoption2",'ipmtemp','doorlock']
ERROR_LIST = ['NAN','UB','DC','DDC']

main_test(db_file_name, sub_output_dir, COLUM_LIST, ERROR_LIST)