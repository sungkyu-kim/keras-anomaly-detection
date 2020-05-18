import demo.RM.information as info
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
TRAIN_RATE = 0.25
DATA_SIZE = 10

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

def create_dataset_time_step(data, value, Xs, Ys, size, error_list):

    ignore_time = 0
    if size < DATA_SIZE :
        print('create_dataset_time_step return , size : '+ str(size))
        return
    is_error_value = 0
    Ydata = []
    for i in range(size):
        error_value = 0
        for j in error_list:
            if (value[i] == j):
                index = error_list.index(value[i])
                #print(' i '+str(i)+' j '+str(j) + ' index ' + intindex)
                error_value = index
                is_error_value = index
        Ydata.append(error_value)
    if is_error_value == 0 :
        print('is_error_value NULL')
        return

    Xs.append(data)
    Ys.append(Ydata)

def create_dataset(df, colum_list, error_list):
    Xs, Ys = [], []

    data = df[colum_list]
    value = df['errorrecode'].values
    di = df['di']

    data = np.where(data < -999, 0, data)
#    scaler = MinMaxScaler()
#    data = scaler.fit_transform(data)

    temp_di = di[0]
    di_count = 0
    start_num = 0
    end_num = 0
    for i in range(len(di)):
        if (temp_di != di[i]) | (i == len(di) - 1):
            end_num = i
            temp_data = data[start_num:end_num]
            temp_value = value[start_num:end_num]

            create_dataset_time_step(temp_data, temp_value, Xs, Ys, di_count, error_list)

            temp_di = di[i]
            start_num = i
            di_count = 1
        else:
            di_count += 1

    return np.array(Xs), np.array(Ys)

def create_dataset_error(data, value, size, error_list):

    if size < DATA_SIZE :
        print('create_dataset_time_step return , size : '+ str(size))
        return
    is_error_value = 0
    Ydata = []
    for i in range(size):
        error_value = 0
        for j in error_list:
            if (value[i] == j):
                index = error_list.index(value[i])
                #print(' i '+str(i)+' j '+str(j) + ' index ' + intindex)
                error_value = index
        Ydata.append(error_value)

    return data, Ydata

def create_dataset_full(df, colum_list, error_list):

    data = df[colum_list]
    value = df['errorrecode'].values

    data = np.where(data < -999, 0, data)
#    scaler = MinMaxScaler()
#    data = scaler.fit_transform(data)

    Xs, Ys = create_dataset_error(data, value, len(value), error_list)

    return np.array(Xs), np.array(Ys)

def create_data(db_file_name, sub_output_dir, colum_list, error_list, is_di):
    output_dir = sub_output_dir + '0_data/'
    if os.path.exists(output_dir):
        print(output_dir + 'Already exist')
    else:
        create_directory(output_dir)

    our_dir_file = output_dir + 'x.npy'
    if os.path.exists(our_dir_file):
        print(our_dir_file + 'Already exist')
        return load_data_XY(output_dir)

    # for file_name in file_list :
    dataframe = pd.read_csv(db_file_name)
    df_size = len(dataframe)

    print('db_file_name : ' + db_file_name)
    print('dataframe.shape and size')
    print(dataframe.shape)
    print(df_size)

    if is_di == True :
        X, Y = create_dataset(dataframe, colum_list, error_list)

        print('X')
        print(X.shape)
        print('Y')
        print(Y.shape)

        error_temp = []
        error_len = len(error_list)

        for i in range(error_len):
            error_temp.append(0)

        print(Y.shape)
        y_len  = len(Y)
        print(y_len)

        for i in range(y_len):
            temp_y = Y[i]
            temp_y_len = len(temp_y)
            for j in range (temp_y_len) :
                error_temp[temp_y[j]] += 1
        print(error_temp)
    else :
        X, Y = create_dataset_full(dataframe, colum_list, error_list)

        print('X')
        print(X.shape)
        print('Y')
        print(Y.shape)

        error_temp = []
        error_len = len(error_list)

        for i in range(error_len):
            error_temp.append(0)

        print(Y.shape)
        y_len = len(Y)
        print(y_len)

        for i in range(y_len):

            for j in range(error_len):
                error_temp[Y[j]] += 1
        print(error_temp)

    info.save_infomation_data(sub_output_dir, db_file_name, dataframe, X, Y, error_temp, colum_list, error_list)
    np.save(output_dir + 'X.npy', X)
    np.save(output_dir + 'Y.npy', Y)

    return X, Y


def load_data(sub_output_dir):
    data_dir = sub_output_dir + '0_data/'
    X = np.load(data_dir + 'X.npy')
    Y = np.load(data_dir + 'Y.npy')

    datasets_dict = {}
    datasets_dict['anomaly'] = (X.copy(), Y.copy())
    return datasets_dict

def load_data_XY(sub_output_dir):
    data_dir = sub_output_dir + '0_data/'
    X = np.load(data_dir + 'X.npy')
    Y = np.load(data_dir + 'Y.npy')

    return X, Y
