import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras_anomaly_detection.library.plot_utils import visualize_reconstruction_error, visualize_anomaly, \
    visualize_anomaly_errors, plot_training_history_file, plot_confusion_matrix_file
from keras_anomaly_detection.library.recurrent import CnnLstmAutoEncoder
from keras_anomaly_detection.library.recurrent import LstmAutoEncoder
from keras_anomaly_detection.library.convolutional import Conv1DAutoEncoder
from keras_anomaly_detection.library.recurrent import BidirectionalLstmAutoEncoder
from keras_anomaly_detection.library.feedforward import FeedForwardAutoEncoder
from keras_anomaly_detection.library.evaluation_utils import report_evaluation_metrics_file
import demo.RM.information as info

import matplotlib.pyplot as plt
import demo.RM.data as data
import time

RANDOM_SEED = 42
WINDOW_SIZE = 5
adjusted_threshold = 2

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

def AutoEncoder_test(X_data, Y_data, sub_output_dir, num, model_name, ae, error_list) :

    model_dir_path = sub_output_dir + model_name + '/'

    anomaly_dir = model_dir_path + 'anomaly/'
    png_dir_1  = sub_output_dir + '1_png/'
    png_dir_2 = model_dir_path + 'png/'
    metrics_dir_1 = sub_output_dir + '2_metrics/'
    metrics_dir_2 = model_dir_path + 'metrics/'
    confusion_dir_1 = sub_output_dir + '3_confusion/'
    confusion_dir_2 = model_dir_path + 'confusion/'

    create_directory(model_dir_path)
    create_directory(anomaly_dir)
    create_directory(png_dir_1)
    create_directory(png_dir_2)
    create_directory(metrics_dir_1)
    create_directory(metrics_dir_2)
    create_directory(confusion_dir_1)
    create_directory(confusion_dir_2)

    x_size = len(X_data)
    y_size = 0
    for i in range (x_size) :
        if Y_data[i] == 0 :
            y_size +=1
    estimated_negative_sample_ratio = y_size / x_size

    # fit the data and save model into model_dir_path
    history = ae.fit(X_data, model_dir_path=model_dir_path, estimated_negative_sample_ratio=estimated_negative_sample_ratio)

    # load back the model saved in model_dir_path detect anomaly
    #ae.load_model(model_dir_path)

    if 0 :
        _, Xtest, _, Ytest = train_test_split(X_data, Y_data, test_size=0.5, random_state=1004)
    else :
        Xtest = X_data
        Ytest = Y_data

    adjusted_threshold = ae.threshold
    anomaly_information = ae.anomaly(Xtest, adjusted_threshold)
    reconstruction_error = []
    Ypred = []

    file_name_info = anomaly_dir + str(num) + '_anomaly.txt'
    f1 = open(file_name_info, mode='at')
    f2 = open(model_dir_path + 'dist.csv', mode='at')

    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        temp_str = '# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')'
        #print(temp_str)
        f1.write(temp_str + '\n')
        index = Y_data[idx]
        predicted_label = 1 if is_anomaly else 0
        Ypred.append(predicted_label)
        reconstruction_error.append(dist)

        anomal_str = str(idx) + ',' + str(index) + ',' + str(dist)
        f2.write(anomal_str+'\n')

    f1.close()
    f2.close()

    png_name_info_1 = png_dir_1 + str(num) + '_' + model_name + '_anomaly.png'
    png_name_info_2 = png_dir_2 + str(num) + '_' + model_name + '_anomaly.png'
    png_title = str(num) + '_'  + model_name + '_' + str(len(X_data))
    visualize_reconstruction_error(reconstruction_error, ae.threshold, Y_data, png_name_info_1, png_name_info_2, png_title, WINDOW_SIZE, error_list)
    plot_training_history_file(history, model_dir_path, num)

    #visualize_anomaly(Ytest, reconstruction_error, adjusted_threshold)
    visualize_anomaly_errors(Ytest, reconstruction_error, adjusted_threshold, error_list, png_title, model_dir_path, num)
    report_evaluation_metrics_file(Ytest, Ypred, metrics_dir_1, metrics_dir_2, num, model_name)
    plot_confusion_matrix_file(Ytest, Ypred, confusion_dir_1, confusion_dir_2, num, model_name)

def main_test(db_file_name, sub_output_dir, COLUM_LIST, ERROR_LIST):
    np.random.seed(RANDOM_SEED)

    if os.path.exists(sub_output_dir):
        print(sub_output_dir + 'Already exist')
    else:
        create_directory(sub_output_dir)

    info.save_infomation(sub_output_dir, date_str, COLUM_LIST, ERROR_LIST)
    X_data, Y_data = data.create_data(db_file_name, sub_output_dir, COLUM_LIST, ERROR_LIST, is_di=False)
    #data_dir_path = './data'
    #model_dir_path = './models'
    #datasets_dict = data.load_data(sub_output_dir)
    #X_data = datasets_dict['anomaly'][0]
    #Y_data = datasets_dict['anomaly'][1]

    X_len = len(X_data)
    x_ = X_data[0]
    y_ = Y_data[0]
    scaler = MinMaxScaler()
    X_scaler_data = scaler.fit_transform(X_data)
    i = 0

    model_name = 'LstmAutoEncoder'
    ae = LstmAutoEncoder()
    AutoEncoder_test(X_scaler_data, Y_data, sub_output_dir, i, model_name, ae, ERROR_LIST)

    model_name = 'CnnLstmAutoEncoder'
    ae = CnnLstmAutoEncoder()
    AutoEncoder_test(X_scaler_data, Y_data, sub_output_dir, i, model_name, ae, ERROR_LIST)

    model_name = 'Conv1DAutoEncoder'
    ae = Conv1DAutoEncoder()
    AutoEncoder_test(X_scaler_data, Y_data, sub_output_dir, i, model_name, ae, ERROR_LIST)

    model_name = 'BidirectionalLstmAutoEncoder'
    ae = BidirectionalLstmAutoEncoder()
    AutoEncoder_test(X_scaler_data, Y_data, sub_output_dir, i, model_name, ae, ERROR_LIST)

    model_name = 'FeedForwardAutoEncoder'
    ae = FeedForwardAutoEncoder()
    AutoEncoder_test(X_scaler_data, Y_data, sub_output_dir, i, model_name, ae, ERROR_LIST)
    info.save_information_done(sub_output_dir)



#if __name__ == '__main__':
#    main()
DATA_DIR = './data/'
RESULT_DIR = './results/2_full/'
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