#@title Imports 
import os
import mne 
import pickle
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt

from Inner_Speech_Dataset.Python_Processing.Data_extractions import  Extract_data_from_subject, Extract_subject_from_BDF, load_events
from Inner_Speech_Dataset.Python_Processing.Data_processing import  Select_time_window, Transform_for_classificator, Split_trial_in_time,  Filter_by_condition


mne.set_log_level(verbose='warning') #to avoid info at terminal
warnings.filterwarnings(action = "ignore", category = DeprecationWarning ) 
warnings.filterwarnings(action = "ignore", category = FutureWarning )

from EEGNet import *
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn import metrics
import tensorflow as tf

### Hyperparameters

# The root dir have to point to the folder that cointains the database
root_dir = "/data/datasets/inner_speech/ds003626/"

# Data Type
datatype = "EEG"

# Sampling rate
fs = 256

# Select the useful par of each trial. Time in seconds
t_start = 1.5
t_end = 3.5

# Subject number
N_S1 = 1
N_S8 = 1

#@title Data extraction and processing

# Load all trials for a sigle subject
X8, Y8 = Extract_data_from_subject(root_dir, N_S8, datatype)



# Cut usefull time. i.e action interval
X8 = Select_time_window(X = X8, t_start = t_start, t_end = t_end, fs = fs)

X8, Y8 = Filter_by_condition(X8, Y8, "INNER")

y_labels8 = Y8[:,1]



kernLength = 106
f1 = 40
f2 = 128
d = 10
dr = 0.5

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

def find_worst_channel(channel_mask, X, y_labels):
    if sum(channel_mask) > 32:

        used_channels = np.array(list(range(128)))
        
        accuracies = []

        for i in range(128):
            if channel_mask[i] == False:
                accuracies.append(float('-inf'))
            else:
                temp_mask = np.copy(channel_mask)
                temp_mask[i] = False
                channels = used_channels[channel_mask]
                print(f'\n\n\n\n Channels: {channels}\n\n\n\n')
                if channels.any():
                    acc = get_av_acc(X, y_labels, channels)
                    accuracies.append(acc)
                else:
                    with open(f'./removing_channels2/ERROR_{sum(channel_mask)}_{i}.txt', 'w+') as f:
                        print(f'\nError No Channels, mask: {channel_mask}\n', file=f)
                        print(f'\nChannels: {channels}\n', file=f)
                        print(f'\nUsed channels: {used_channels}\n', file=f)
        
        worst_chan_1 = accuracies.index(max(accuracies))
        temp_accuracies = accuracies
        temp_accuracies[worst_chan_1] = float('inf')
        worst_chan_2 = temp_accuracies.index(max(temp_accuracies))

        channel_mask[worst_chan_1] = False
        channel_mask[worst_chan_2] = False
        
        real_accuracies = np.array(accuracies)[channel_mask]
        
        with open(f'./removing_channels2/Removing_Channels_Output_{sum(channel_mask)}.txt', 'w+') as f:
            print(f'\nNumber Channels: {sum(channel_mask)}\n', file=f)
            print(f'\nWorst Channel1: {worst_chan_1}\n', file=f)
            print(f'\nWorst Channel2: {worst_chan_2}\n', file=f)
            print('\nAccuracies:\n', file=f)
            print(real_accuracies, file=f)
            print('\nAvg. Accuracy:\n', file=f)
            print(sum(real_accuracies)/len(real_accuracies), file=f)
            print('\nMask:\n', file=f)
            print(channel_mask, file=f)
            print('\Accuracies:\n', file=f)
            print(accuracies, file=f)
            print('\n\n\n', file=f)
        
        
        return find_worst_channel(channel_mask, X, y_labels)
        
    else:
        return channel_mask


    

def get_av_acc(X, y_labels, channels):
    accuracies = []
    
    X = X[:,channels,:]
    
    kf = KFold(n_splits=4, shuffle=True)
    kf.get_n_splits(X)

    kernels, chans, samples = 1, X.shape[1], X.shape[2]

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y_labels[train_index], y_labels[test_index]
        # convert labels to one-hot encodings.
        Y_train      = np_utils.to_categorical(Y_train)
        Y_test       = np_utils.to_categorical(Y_test)
        X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

        with strategy.scope():

            model = EEGNet(nb_classes = 4, Chans = chans, Samples = samples, 
                       dropoutRate = 0.5, kernLength = 64, F1 = 32, D = 10, F2 = 50, 
                       dropoutType = 'Dropout')
            model.compile(
                    tf.keras.optimizers.Adam(
                        name="Adam"),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])


        fittedModel = model.fit(X_train, Y_train, batch_size = 32, epochs = 200, 
                            verbose = 2,
                            callbacks=[callback])

        probs       = model.predict(X_test)
        preds       = probs.argmax(axis = -1)  
        acc         = np.mean(preds == Y_test.argmax(axis=-1))
        accuracies.append(acc)
        print(acc)
    
    return sum(accuracies)/4
    


chan_mask = np.array(list(map(lambda x: True, list(range(128)))))

best_mask = find_worst_channel(chan_mask, X8, y_labels8)

print(best_mask)