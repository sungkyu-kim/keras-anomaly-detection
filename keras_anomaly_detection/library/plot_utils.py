from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

LABELS = ["Normal", "Fraud"]
#https://matplotlib.org/examples/color/named_colors.html
cmap_list = ['magenta', 'darkred', 'coral', 'tomato', 'firebrick', 'orangered', 'brown', 'darkviolet', 'peru']

def plot_confusion_matrix(y_true, y_pred):
    plt.clf()
    conf_matrix = confusion_matrix(y_true, y_pred)

    fit = plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    #plt.show()
    fig.close(fig)

def plot_confusion_matrix_file(y_true, y_pred, confusion_dir_1, confusion_dir_2, png_title):
    plt.clf()
    output_file_1 = confusion_dir_1 + png_title + '_confusion.png'
    output_file_2 = confusion_dir_2 + png_title + '_confusion.png'

    y_true_, y_pred_ = [], []
    for i in range (len(y_true)) :
        if y_true[i] > 0 :
            y_true_.append(1)
        else :
            y_true_.append(0)
        if y_pred[i] > 0 :
            y_pred_.append(1)
        else :
            y_pred_.append(0)

    conf_matrix = confusion_matrix(y_true_, y_pred_)

    fig = plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title(png_title)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig(output_file_1)
    if confusion_dir_2 != 'NULL'
        plt.savefig(output_file_2)
    #plt.show()
    fig.close(fig)

def plot_confusion_matrix_file_threshold(y_true, y_pred, confusion_dir_1, threshold, png_title):
    
    output_file_1 = confusion_dir_1 + png_title + str(threshold) + '_confusion.png'

    plt.clf()
    y_true_, y_pred_ = [], []
    for i in range (len(y_true)) :
        if y_true[i] > 0 :
            y_true_.append(1)
        else :
            y_true_.append(0)
        if y_pred[i] > 0 :
            y_pred_.append(1)
        else :
            y_pred_.append(0)

    conf_matrix = confusion_matrix(y_true_, y_pred_)

    fig = plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title(png_title)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig(output_file_1)
    #plt.show()
    fig.close(fig)

def plot_training_history(history):
    if history is None:
        return
    plt.clf()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()


def visualize_anomaly(y_true, reconstruction_error, threshold):
    error_df = pd.DataFrame({'reconstruction_error': reconstruction_error,
                             'true_class': y_true})
    print(error_df.describe())

    plt.clf()
    groups = error_df.groupby('true_class')
    fig, ax = plt.subplots()

    for name, group in groups:
        ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
                label="Fraud" if name == 1 else "Normal")

    ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    ax.legend(loc='upper left')
    plt.title("Reconstruction error for different classes")
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data point index")
    #plt.show()

def visualize_reconstruction_error(Y_data, reconstruction_error, threshold, gap_dir_1, gap_dir_2, train_length, png_title, error_list):
    plt.clf()

    gap_name_info_1 = gap_dir_1 + png_title + '_gap.png'
    gap_name_info_2 = gap_dir_2 + png_title + '_gap.png'

    plt.plot(reconstruction_error, marker='o', ms=3.5, linestyle='', label='Point')

    error_name = ["Normal"]
    for i in range(1, len(error_list)) :
        error_name.append(error_list[i])

    plt.hlines(threshold, xmin=0, xmax=len(reconstruction_error)-1, colors="b", zorder=100, label='Threshold')
    y_size = len(Y_data)
    ymax = max(reconstruction_error)
    for j in range(1, len(error_list)):
        x_list = []
        is_error = False
        for i in range (train_length, y_size) :
            if Y_data[i] == j :
                x_list.append(i)

                is_error = True
                #plt.text(i+1, ymax/2+0.1, error_txt)
        if is_error == True :
            plt.vlines(x=x_list, ymin=0, ymax=ymax,colors=cmap_list[j], label=error_list[j])
    
    plt.legend(loc='upper left')
    plt.title(png_title)
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data point index")
    plt.grid()
    plt.savefig(gap_name_info_1)
    plt.savefig(gap_name_info_2)
    #plt.show()

def visualize_anomaly_errors(Y_data, reconstruction_error, threshold, anomaly_dir_1, anomaly_dir_2, train_length, png_title, error_list):
    anomaly_name_info_1 = anomaly_dir_1 + png_title + '_anomaly.png'
    anomaly_name_info_2 = anomaly_dir_2 + png_title + '_anomaly.png'
    
    error_df = pd.DataFrame({'reconstruction_error': reconstruction_error, 'true_class': Y_data})
    print(error_df.describe())

    plt.clf()
    groups = error_df.groupby('true_class')
    fig, ax = plt.subplots()

    error_name = ["Normal"]
    for i in range(1, len(error_list)) :
        error_name.append(error_list[i])

    for name, group in groups:
        ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='', label=error_name[name])

    ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    ax.legend(loc='upper left')
    #plt.title("Reconstruction error for different classes")
    plt.title(png_title)
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data point index")
    plt.savefig(anomaly_name_info_1)
    plt.savefig(anomaly_name_info_2)    
    #plt.show()
    plt.close(fig)

def plot_training_history_file(history, loss_dir_1, loss_dir_2, num, model_name):
    if history is None:
        return
    plt.clf()
    loss_dir_1_str = loss_dir_1 + 'loss_'+str(num) + '_' + model_name + '.png'
    loss_dir_2_str = loss_dir_2 + 'loss_'+str(num) + '_' + model_name + '.png'
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(loss_dir_1_str)
    plt.savefig(loss_dir_2_str)
    #plt.show()
