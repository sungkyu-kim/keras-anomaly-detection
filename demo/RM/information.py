import time
import os
import matplotlib.pyplot as plt

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

start_time = 0
def save_infomation(output_dir, date_str, colum_list, error_list) :
    info_file_name = output_dir +'information.txt'
    start_time = time.time()
    f = open(info_file_name, mode='wt')
    f.write('\n\n\n <<< save_information >>> ')
    f.write('\ndate_str : ' + date_str)
    f.write('\ncolum_list : ' + str(colum_list))
    f.write('\nerror_list : ' + str(error_list))
    f.write('\nstart_time : ' + str(start_time))
    f.close()
    return

def save_infomation_data(output_dir, db_file_name, dataframe, X, Y, error_temp, colum_list, error_list):
    info_file_name = output_dir + 'information.txt'

    f = open(info_file_name, mode='at')

    f.write('\n\n\n <<< save_information_data >>> ')
    f.write('\n db_file_name : ' + db_file_name)
    f.write('\n dataframe.shape and size')
    f.write(str(dataframe.shape))
    f.write('\ncolum_list : ' + str(colum_list))
    f.write('\nerror_list : ' + str(error_list))
    f.write('\n X : ' + str(X.shape))
    f.write('\n Y : ' + str(Y.shape))
    f.write('\n error_temp : ' + str(error_temp))
    f.close()

    range_file_name = output_dir +'data_range.txt'
    f = open(range_file_name, mode='wt')
    x_size = len(X)
    xx_list = []
    for i in range (x_size) :
        xx_size = len(X[i])
        xx_list.append(xx_size)
        temp_str = '[' + str(i) + '] : ' + str(xx_size) + '\n'
        f.write(temp_str)
    f.close()

    range_png_name = output_dir + 'data_range.png'
    plt.plot(xx_list)
    plt.grid()
    plt.savefig(range_png_name)
    return

def save_information_done(output_dir):
    info_file_name = output_dir + 'information.txt'
    f = open(info_file_name, mode='at')
    f.write('\n\n\n <<< save_information_done >>> ')
    end_date_str = time.strftime("%m%d_%H%M")
    f.write('\n end_date_str : ' + end_date_str)
    duration = time.time() - start_time
    f.write('\n duration : ' + str(duration))
    f.close()
