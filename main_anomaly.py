import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras_anomaly_detection.library.plot_utils import visualize_reconstruction_error, plot_training_history_file, \
    plot_confusion_matrix_file, visualize_anomaly_errors, plot_confusion_matrix_file_threshold
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
DO_TRAINING = False
RANDOM_SEED = 42

adjusted_threshold = 2
threshold_list = [[0.04,0.05,0.06,0.07,0.08,0.09],[0.08,0.1,0.12,0.14,0.16],[0.3,0.5,0.7,0.9,0.11],[0.3,0.5,0.7,0.9,0.11]]

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

def AutoEncoder_test_train(X_data, sub_output_dir, num, model_name, ae, error_list, test_type, train_length) :
    
    model_dir_path = sub_output_dir + model_name + '/'
    
    loss_dir_1 = sub_output_dir + '5_loss/'
    loss_dir_2 = model_dir_path + '5_loss/'

    create_directory(loss_dir_1)
    create_directory(loss_dir_2)
    
    # fit the data and save model into model_dir_path
    history = ae.fit(X_data, model_dir_path=model_dir_path)

    plot_training_history_file(history, loss_dir_1, loss_dir_2, num)

def AutoEncoder_test_pred(X_data, Y_data, sub_output_dir, num, model_name, ae, error_list, test_type, train_length, threshold) :
    
    model_dir_path = sub_output_dir + model_name + '/'
    
    gap_dir_1  = sub_output_dir + '1_gap/' + model_name + '/' + str(threshold) + '/'
    gap_dir_2 = model_dir_path + '1_gap/' + model_name + '/' + str(threshold) + '/'
    anomaly_dir_1 = sub_output_dir + '2_anomaly/' + model_name + '/' + str(threshold) + '/'
    anomaly_dir_2 = model_dir_path + '2_anomaly/' + model_name + '/' + str(threshold) + '/'
    confusion_dir_1 = sub_output_dir + '3_confusion/' + model_name + '/' + str(threshold) + '/'
    confusion_dir_2 = model_dir_path + '3_confusion/' + model_name + '/' + str(threshold) + '/'
    metrics_dir_1 = sub_output_dir + '4_metrics/' + model_name + '/' + str(threshold) + '/'
    metrics_dir_2 = model_dir_path + '4_metrics/' + model_name + '/' + str(threshold) + '/'
    dist_dir_1 = sub_output_dir + '6_dist/' + model_name + '/' + str(threshold) + '/'
    dist_dir_2 = model_dir_path + '6_dist/' + model_name + '/' + str(threshold) + '/'
    csv_dir_1 = sub_output_dir + '7_csv/' + model_name + '/' + str(threshold) + '/'
    csv_dir_2 = model_dir_path + '7_csv/' + model_name + '/' + str(threshold) + '/'
    csv_dir_3 = sub_output_dir + '7_csv_dist/' + model_name + '/' + str(threshold) + '/'

    create_directory(model_dir_path)
    create_directory(gap_dir_1)
    create_directory(gap_dir_2)
    create_directory(anomaly_dir_1)
    create_directory(anomaly_dir_2)
    create_directory(confusion_dir_1)
    create_directory(confusion_dir_2)
    create_directory(metrics_dir_1)
    create_directory(metrics_dir_2)
    create_directory(dist_dir_1)
    create_directory(dist_dir_2)
    create_directory(csv_dir_1)
    create_directory(csv_dir_2)
    create_directory(csv_dir_3)

    # load back the model saved in model_dir_path detect anomaly
    #ae.load_model(model_dir_path)
    anomaly_information = ae.anomaly(X_data, threshold=estimated_negative_sample_ratio)
    reconstruction_error = []
    
    dist_dir_1_str = dist_dir_1 + str(num) + '_' + model_name + '_dist.txt'
    dist_dir_2_str = dist_dir_2 + str(num) + '_' + model_name + '_dist.txt'
    dist_dir_1_file = open(dist_dir_1_str, mode='at')
    dist_dir_2_file = open(dist_dir_2_str, mode='at')

    csv_dir_1_str = csv_dir_1 + str(num) + '_' + model_name + '_dist.csv'
    csv_dir_2_str = csv_dir_2 + str(num) + '_' + model_name + '_dist.csv'
    csv_dir_3_str_abnomal = csv_dir_3 + model_name + '_dist_abnormal.csv'
    csv_dir_3_str_normal = csv_dir_3 + model_name + '_dist_normal.csv'
    csv_dir_1_file = open(csv_dir_1_str, mode='at')
    csv_dir_2_file = open(csv_dir_2_str, mode='at')
    csv_dir_3_file_abnormal = open(csv_dir_3_str_abnomal, mode='at')
    csv_dir_3_file_normal = open(csv_dir_3_str_normal, mode='at')

    Ypred = []
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        temp_str = '# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')'
        #print(temp_str)
        dist_dir_1_file.write(temp_str + '\n')
        dist_dir_2_file.write(temp_str + '\n')

        index = Y_data[idx]

        predicted_label = 1 if is_anomaly else 0
        Ypred.append(predicted_label)
        reconstruction_error.append(dist)

        anomal_str = str(idx) + ',' + str(index) + ',' + str(dist)
        csv_dir_1_file.write(anomal_str+'\n')
        csv_dir_2_file.write(anomal_str+'\n')
        if index == 0 :
            csv_dir_3_file_normal.write(str(dist) + '\n')
        else :
            csv_dir_3_file_abnormal.write(str(dist) + '\n')
            csv_dir_3_str_abnoraml_no = csv_dir_3 + model_name + '_dist_abnormal_' + str(index) + '.csv'
            csv_dir_3_file_abnormal_no = open(csv_dir_3_str_abnoraml_no, mode='at')
            csv_dir_3_file_abnormal_no.write(str(dist) + '\n')
            csv_dir_3_file_abnormal_no.close()

    dist_dir_1_file.close()
    dist_dir_2_file.close()
    csv_dir_1_file.close()
    csv_dir_2_file.close()    
    csv_dir_3_file_normal.close()
    csv_dir_3_file_abnormal.close()
    
    adjusted_threshold = ae.threshold

    png_title = str(num) + '_' + model_name + '_' + str(len(X_data))
    visualize_reconstruction_error(Y_data, reconstruction_error, adjusted_threshold, gap_dir_1, gap_dir_2, train_length, png_title, error_list)
    visualize_anomaly_errors(Y_data, reconstruction_error, adjusted_threshold, anomaly_dir_1, anomaly_dir_2, train_length, png_title, error_list)
    plot_confusion_matrix_file(Y_data, Ypred, confusion_dir_1, confusion_dir_2, png_title)
    report_evaluation_metrics_file(Y_data, Ypred, metrics_dir_1, metrics_dir_2, png_title, model_name)
    return Y_data, Ypred

def AutoEncoder_test(X_data, Y_data, sub_output_dir, num, model_name, ae, error_list, test_type, train_length) :

    model_dir_path = sub_output_dir + model_name + '/'
    
    gap_dir_1  = sub_output_dir + '1_gap/'
    gap_dir_2 = model_dir_path + '1_gap/'
    anomaly_dir_1 = sub_output_dir + '2_anomaly/'
    anomaly_dir_2 = model_dir_path + '2_anomaly/'
    confusion_dir_1 = sub_output_dir + '3_confusion/'
    confusion_dir_2 = model_dir_path + '3_confusion/'
    metrics_dir_1 = sub_output_dir + '4_metrics/'
    metrics_dir_2 = model_dir_path + '4_metrics/'
    loss_dir_1 = sub_output_dir + '5_loss/'
    loss_dir_2 = model_dir_path + '5_loss/'
    dist_dir_1 = sub_output_dir + '6_dist/'
    dist_dir_2 = model_dir_path + '6_dist/'
    csv_dir_1 = sub_output_dir + '7_csv/'
    csv_dir_2 = model_dir_path + '7_csv/'

    create_directory(model_dir_path)
    create_directory(gap_dir_1)
    create_directory(gap_dir_2)
    create_directory(anomaly_dir_1)
    create_directory(anomaly_dir_2)
    create_directory(confusion_dir_1)
    create_directory(confusion_dir_2)
    create_directory(metrics_dir_1)
    create_directory(metrics_dir_2)
    create_directory(loss_dir_1)
    create_directory(loss_dir_2)
    create_directory(dist_dir_1)
    create_directory(dist_dir_2)
    
    x_size = len(X_data)
    y_size = 0
    for i in range (x_size) :
        if Y_data[i] == 0 :
            y_size +=1
    estimated_negative_sample_ratio = y_size / x_size

    if test_type == 1 :
        x_train = X_data
        y_train = Y_data
        x_test = X_data
        y_test = Y_data
    elif test_type == 2 :
        x_train = X_data[:train_length]
        y_train = Y_data[:train_length]
        x_test = X_data[train_length:]
        y_test = Y_data[train_length:]
    elif test_type == 3 :
        x_train = X_data[:train_length]
        y_train = Y_data[:train_length]
        x_test = X_data
        y_test = Y_data
    elif test_type == 4 :
        x_train = X_data[:train_length]
        y_train = Y_data[:train_length]
        x_test = X_data
        y_test = Y_data
    
    # fit the data and save model into model_dir_path
    history = ae.fit(x_train, model_dir_path=model_dir_path, estimated_negative_sample_ratio=estimated_negative_sample_ratio)

    # load back the model saved in model_dir_path detect anomaly
    #ae.load_model(model_dir_path)
    anomaly_information = ae.anomaly(x_test)
    reconstruction_error = []
    
    dist_dir_1_str = dist_dir_1 + str(num) + '_' + model_name + '_dist.txt'
    dist_dir_2_str = dist_dir_2 + str(num) + '_' + model_name + '_dist.txt'
    dist_dir_1_file = open(dist_dir_1_str, mode='at')
    dist_dir_2_file = open(dist_dir_2_str, mode='at')

    csv_dir_1_str = csv_dir_1 + str(num) + '_' + model_name + '_dist.csv'
    csv_dir_2_str = csv_dir_2 + str(num) + '_' + model_name + '_dist.csv'
    csv_dir_1_file = open(csv_dir_1_str, mode='at')
    csv_dir_2_file = open(csv_dir_2_str, mode='at')

    Ypred = []
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        temp_str = '# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')'
        #print(temp_str)
        dist_dir_1_file.write(temp_str + '\n')
        dist_dir_2_file.write(temp_str + '\n')

        index = y_test[idx]

        predicted_label = 1 if is_anomaly else 0
        Ypred.append(predicted_label)
        reconstruction_error.append(dist)

        anomal_str = str(idx) + ',' + str(index) + ',' + str(dist)
        csv_dir_1_file.write(anomal_str+'\n')
        csv_dir_2_file.write(anomal_str+'\n')

    dist_dir_1_file.close()
    dist_dir_2_file.close()
    csv_dir_1_file.close()
    csv_dir_2_file.close()
    
    adjusted_threshold = ae.threshold

    png_title = str(num) + '_' + model_name + '_' + str(len(X_data))
    visualize_reconstruction_error(y_test, reconstruction_error, adjusted_threshold, gap_dir_1, gap_dir_2, train_length, png_title, error_list)
    visualize_anomaly_errors(y_test, reconstruction_error, adjusted_threshold, anomaly_dir_1, anomaly_dir_2, train_length, png_title, error_list)
    plot_confusion_matrix_file(y_test, Ypred, confusion_dir_1, confusion_dir_2, png_title)

    plot_training_history_file(history, loss_dir_1, loss_dir_2, num, model_name)    
    report_evaluation_metrics_file(y_test, Ypred, metrics_dir_1, metrics_dir_2, png_title, model_name)
    return y_test, Ypred


def main_test(db_file_name, Test_dir, date_str, COLUM_LIST, ERROR_LIST, test_type, train_length):

    np.random.seed(RANDOM_SEED)

    create_directory(Test_dir)

    info.save_infomation(Test_dir, Test_dir, COLUM_LIST, ERROR_LIST)
    X_data, Y_data = data.create_data(db_file_name, Test_str, COLUM_LIST, ERROR_LIST, test_type, train_length)
    #data_dir_path = './data'
    #model_dir_path = './models'
    #datasets_dict = data.load_data(sub_output_dir)
    #X_data = datasets_dict['anomaly'][0]
    #Y_data = datasets_dict['anomaly'][1]

    X_len = len(X_data)
    Lstm_y_test, Lstm_y_pred = [], []
    CnnLstm_y_test, CnnLstm_y_pred = [], []
    Conv1D_y_test, Conv1D_y_pred = [], []
    BidirectionalLstm_y_test, BidirectionalLstm_y_pred = [], []

    for i in range (X_len) :
        x_ = X_data[i]
        y_ = Y_data[i]
        scaler = MinMaxScaler()
        X_scaler_data = scaler.fit_transform(x_)

        model_name = 'LstmAutoEncoder'
        print('< %s > [%d / %d]'%(model_name,i,X_len))
        ae = LstmAutoEncoder()
        y_test, y_pred, = AutoEncoder_test(X_scaler_data, y_, Test_dir, i, model_name, ae, ERROR_LIST, test_type, train_length)
        Lstm_y_test.append(y_test)
        Lstm_y_pred.append(y_pred)

        model_name = 'CnnLstmAutoEncoder'
        print('< %s > [%d / %d]'%(model_name,i,X_len))
        ae = CnnLstmAutoEncoder()
        y_test, y_pred, = AutoEncoder_test(X_scaler_data, y_, Test_dir, i, model_name, ae, ERROR_LIST, test_type, train_length)
        CnnLstm_y_test.append(y_test)
        CnnLstm_y_pred.append(y_pred)

        model_name = 'Conv1DAutoEncoder'
        print('< %s > [%d / %d]'%(model_name,i,X_len))
        ae = Conv1DAutoEncoder()
        y_test, y_pred, = AutoEncoder_test(X_scaler_data, y_, Test_dir, i, model_name, ae, ERROR_LIST, test_type, train_length)
        Conv1D_y_test.append(y_test)
        Conv1D_y_pred.append(y_pred)

        model_name = 'BidirectionalLstmAutoEncoder'
        print('< %s > [%d / %d]'%(model_name,i,X_len))
        ae = BidirectionalLstmAutoEncoder()
        y_test, y_pred, = AutoEncoder_test(X_scaler_data, y_, Test_dir, i, model_name, ae, ERROR_LIST, test_type, train_length)
        BidirectionalLstm_y_test.append(y_test)
        BidirectionalLstm_y_pred.append(y_pred)

    metrics_dir = Test_dir + 'total_metrics/'
    create_directory(metrics_dir)
    confusion_dir = Test_dir + 'total_plot/'
    create_directory(confusion_dir)

    model_name = 'LstmAutoEncoder'    
    y_test , y_pred = [], []
    for i in range (len(Lstm_y_test)) :
        temp_y_test = Lstm_y_test[i]
        temp_y_pred = Lstm_y_pred[i]
        for j in range (len(temp_y_test)) :
            y_test.append(temp_y_test[j])
            y_pred.append(temp_y_pred[j])
    report_evaluation_metrics_file(y_test, y_pred, metrics_dir, "NULL", model_name, model_name)
    plot_confusion_matrix_file(y_test, y_pred, confusion_dir, "NULL", model_name)

    model_name = 'CnnLstmAutoEncoder'
    y_test , y_pred = [], []
    for i in range (len(CnnLstm_y_test)) :
        temp_y_test = CnnLstm_y_test[i]
        temp_y_pred = CnnLstm_y_pred[i]
        for j in range (len(temp_y_test)) :
            y_test.append(temp_y_test[j])
            y_pred.append(temp_y_pred[j])
    report_evaluation_metrics_file(y_test, y_pred, metrics_dir, "NULL", model_name, model_name)
    plot_confusion_matrix_file(y_test, y_pred, confusion_dir, "NULL", model_name)

    model_name = 'Conv1DAutoEncoder'
    y_test , y_pred = [], []
    for i in range (len(Conv1D_y_test)) :
        temp_y_test = Conv1D_y_test[i]
        temp_y_pred = Conv1D_y_pred[i]
        for j in range (len(temp_y_test)) :
            y_test.append(temp_y_test[j])
            y_pred.append(temp_y_pred[j])
    report_evaluation_metrics_file(y_test, y_pred, metrics_dir, "NULL", model_name, model_name)
    plot_confusion_matrix_file(y_test, y_pred, confusion_dir, "NULL", model_name)

    model_name = 'BidirectionalLstmAutoEncoder'
    y_test , y_pred = [], []
    for i in range (len(BidirectionalLstm_y_test)) :
        temp_y_test = BidirectionalLstm_y_test[i]
        temp_y_pred = BidirectionalLstm_y_pred[i]
        for j in range (len(temp_y_test)) :
            y_test.append(temp_y_test[j])
            y_pred.append(temp_y_pred[j])
    report_evaluation_metrics_file(y_test, y_pred, metrics_dir, "NULL", model_name, model_name)
    plot_confusion_matrix_file(y_test, y_pred, confusion_dir, "NULL", model_name)

    info.save_information_done(Test_dir)

def main_test_train_all_start(model_no, x_train, y_data, model_name, ae, Test_dir, ERROR_LIST, test_type, train_length):
    y_len = len(y_data)
    Lstm_y_test, Lstm_y_pred = [], []

    AutoEncoder_test_train(np.array(x_train), Test_dir, 0, model_name, ae, ERROR_LIST, test_type, train_length)

    for k in (threshold_list[model_no]) :
        for i in range (y_len) :
            x_test_np = np.array(x_test[i])
            y_test_np = np.array(y_data[i])
            y_test, y_pred, = AutoEncoder_test_pred(x_test, y_test, Test_dir, i, model_name, ae, ERROR_LIST, test_type, train_length, k)
            for j in range(len(y_test)) :
                Lstm_y_test.append(y_test[i])
                Lstm_y_pred.append(y_data[i])    
        metrics_dir_sub = Test_dir + model_name + '/'
        title = model_name + '_' + str(k)
        report_evaluation_metrics_file(Lstm_y_test, Lstm_y_pred, Test_dir, metrics_dir_sub, title, model_name)
        plot_confusion_matrix_file_threshold(Lstm_y_test, Lstm_y_pred, Test_dir, k, model_name)

    info.save_information_done(Test_dir)

def main_test_train_all(db_file_name, Test_dir, date_str, COLUM_LIST, ERROR_LIST, test_type, train_length):
    
    np.random.seed(RANDOM_SEED)

    create_directory(Test_dir)

    info.save_infomation(Test_dir, Test_dir, COLUM_LIST, ERROR_LIST)
    X_data, Y_data = data.create_data(db_file_name, Test_dir, COLUM_LIST, ERROR_LIST, test_type, train_length)
    #data_dir_path = './data'
    #model_dir_path = './models'
    #datasets_dict = data.load_data(sub_output_dir)
    #X_data = datasets_dict['anomaly'][0]
    #Y_data = datasets_dict['anomaly'][1]

    X_len = len(X_data)

    x_list, x_list_count = [], []
    x_test, y_test = [], []

    for i in range (X_len) :        
        x_temp = X_data[i]
        x_temp_len = len(x_temp)
        for j in range (x_temp_len) :
            x_list.append(x_temp[j])
        x_list_count.append(x_temp_len)

    scaler = MinMaxScaler()
    X_scaler_data = scaler.fit_transform(x_list)

    count = 0
    for i in range (X_len) :
        temp_len = x_list_count[i]
        for j in range (train_length) :
            x_train.append(X_scaler_data[count+j])
        x_test.append(X_scaler_data[count:count+temp_len])
        y_test.append(Y_data[i])
        count += temp_len

    metrics_dir = Test_dir

    model_name = 'LstmAutoEncoder'
    ae = LstmAutoEncoder()
    main_test_train_all_start(0, x_train, x_test, Y_data, model_name, ae, Test_dir, ERROR_LIST, test_type, train_length)

    model_name = 'CnnLstmAutoEncoder'
    ae = CnnLstmAutoEncoder()
    main_test_train_all_start(1, x_train, x_test, Y_data, model_name, ae, Test_dir, ERROR_LIST, test_type, train_length)

    model_name = 'Conv1DAutoEncoder'
    ae = Conv1DAutoEncoder()
    main_test_train_all_start(2, x_train, x_test, Y_data, model_name, ae, Test_dir, ERROR_LIST, test_type, train_length)

    model_name = 'BidirectionalLstmAutoEncoder'
    ae = BidirectionalLstmAutoEncoder()
    main_test_train_all_start(3, x_train, x_test, Y_data, model_name, ae, Test_dir, ERROR_LIST, test_type, train_length)

    info.save_information_done(Test_dir)


#if __name__ == '__main__':
#    main()

#test_type = 0  full
#test_type = 1  di
#test_type = 2  init di
#test_type = 3  init di all
#test_type = 4  init train all

DATA_DIR = './demo/RM/data/'
RESULT_DIR = './Results/'
db_file_name = 'ub.csv'
date_str = time.strftime("%m%d_%H%M")

Result_Test_Dir = RESULT_DIR + date_str + '/'
data.create_directory(Result_Test_Dir)

test_name = '1_Init_test_2_500'
COLUM_LIST = ["courseoption1","courseoption2",'ipmtemp','doorlock']
ERROR_LIST = ['NAN','UB','DC','DDC']
date_str = time.strftime("%m%d_%H%M")
Test_dir = Result_Test_Dir + date_str + '_' + test_name + '/'
test_type = 2
train_length = 300
main_test(DATA_DIR+db_file_name, Test_dir, date_str, COLUM_LIST, ERROR_LIST, test_type, train_length)


test_name = '2_Init_train_all_test_3_500'
COLUM_LIST = ["courseoption1","courseoption2",'ipmtemp','doorlock']
ERROR_LIST = ['NAN','UB','DC','DDC']
date_str = time.strftime("%m%d_%H%M")
Test_dir = Result_Test_Dir + date_str + '_' + test_name + '/'
test_type = 3
train_length = 500
main_test_train_all(DATA_DIR+db_file_name, Test_dir, date_str, COLUM_LIST, ERROR_LIST, test_type, train_length)