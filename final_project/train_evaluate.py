# +
#@title Imports 
# %pip install keras-tuner --upgrade
import keras_tuner as kt

import os
import mne 
import pickle
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
# -

from Inner_Speech_Dataset.Python_Processing.Data_extractions import  Extract_data_from_subject
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
import json




# ## Hyperparameters

# The root dir have to point to the folder that cointains the database
root_dir = "/data/datasets/inner_speech/ds003626/"

# Data Type
datatype = "EEG"

# Sampling rate
fs = 256

# Select the useful par of each trial. Time in seconds
t_start = 1.5
t_end = 3.5


# +
# kernLength = 114
# f1 = 28
# f2 = 74
# d = 18
# dr = 0.55

av_accuracies = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
av_precisions = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
av_recalls = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
av_F1s = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}


# -

def build_model(hp):  
    """
    Method used by the tuner to build models
    param: hp: the hyperparameters
    """
    # Create the keras model
    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
    # model configurations may do better, but this is a good starting point)

    kernLength = hp.Int(
                    "kernel_length",
                    16,
                    100,
                    step=2
                )

    temporal_filters = hp.Int(
                    "temporal_filters",
                    20,
                    72,
                    step=4
                )

    pointwise_filters = hp.Int(
                    "pointwise_filters",
                    28,
                    136,
                    step=4
                )

    spatial_filters = hp.Int(
                    "spatial_filters",
                    2,
                    32,
                    step=2
                )

    

    model = EEGNet(nb_classes = 4, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = kernLength, 
               F1 = temporal_filters, D = spatial_filters, F2 = pointwise_filters, 
               dropoutType = 'Dropout')

    
    
    # Convert model to run on multiple GPUs
    #parallel_model = multi_gpu_model(model, gpus=8)
    
    model.compile(
        tf.keras.optimizers.Adam(
            # Specifies the range of values for the tuner to try
           # learning_rate= hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log"),
            name="Adam"),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

# +
# Subject number
for subject in range(1,9):
    # Load all trials for a sigle subject
    X, Y = Extract_data_from_subject(root_dir, subject, datatype)

    # Cut usefull time. i.e action interval
    X = Select_time_window(X = X, t_start = t_start, t_end = t_end, fs = fs)

    X, Y = Filter_by_condition(X, Y, "INNER")


    # left_channels = np.array([0,
    #                     1,
    #                     3,
    #                     4,
    #                     7,
    #                     8,
    #                     9,
    #                     14,
    #                     18,
    #                     19,
    #                     21,
    #                     22,
    #                     23,
    #                     81,
    #                     83,
    #                     90,
    #                     94,
    #                     95,
    #                     100,
    #                     101,
    #                     105,
    #                     110,
    #                     114,
    #                     116,
    #                     121,
    #                     125
    #                     ])

    best_channel_mask = [False, False,  True,  True, False, True, False, False, False, False, False,  True,
        False,  True, False, False, False, False, False, False, False, False,  True, False,
        False, False, False, False, False, False, False,  True, False, False, False, False,
        False,  True, False, False,  True, False,  True, False,  True,  True, False, False,
        False, False, False, False, False, False, False,  True, False, False, False, False,
        False,  True, False, False, False,  True, False, False, False, False,  True,  True,
        False, False,  True, False, False, False, False, False, False, False,  True, False,
        False,  True, False,  True, False, False,  True,  True, False, False, False, False,
        True,  True, False, False, False,  True, False, False,  True, False, False, False,
        False, False,  True, False, False, False,  True, False,  True, False, False, False,
        False, False, False,  True, False,  True, False, False,]

    X = X[:,best_channel_mask,:]

    x_min = X.min(axis=(2), keepdims=True)
    x_max = X.max(axis=(2), keepdims=True)

    x_min.shape


    # Normalize data
    X = (X - x_min)/(x_max-x_min)

    y_labels = Y[:,1]

    kf = KFold(n_splits=4, shuffle=True)
    kf.get_n_splits(X)
    accuracies = []

    kernels, chans, samples = 1, X.shape[1], X.shape[2]

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


    accuracies = []
    precisions = []
    recalls = []
    F1s = []

    print(f'\n\n\n\nNEW TUNING FOR SUBJECT: {subject}\n\n\n\n')

    tuner = kt.Hyperband(
                build_model,
                objective='val_accuracy',
                max_epochs=300,
                hyperband_iterations=2,
                overwrite=True,
                factor=10,
                distribution_strategy=strategy,
                directory="search_results",
                project_name=f'EEGNet_tuning_subject{subject}'
        )

    Y_temp      = np_utils.to_categorical(y_labels)
    X_temp     = X.reshape(X.shape[0], chans, samples, kernels)

    tuner.search(X_temp,
                Y_temp,
                validation_split=0.35,
                epochs=300,
                verbose=2)

    best_model = tuner.get_best_models(1)[0]
    best_model.save(f'./Best_chan_20/best_EEGNet_S{subject}.h5')
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    summary = tuner.results_summary()

    with open(f'./Best_chan_20/best_EEGNet_Subject_{subject}.txt', 'w+') as f:
        print(f'Best kernel_length: {best_hyperparameters.get("kernel_length")}', file=f)
        print(f'Best spatial_filters: {best_hyperparameters.get("spatial_filters")}', file=f)
        print(f'Best temporal_filters: {best_hyperparameters.get("temporal_filters")}', file=f)
        print(f'Best pointwise_filters: {best_hyperparameters.get("pointwise_filters")}', file=f)
        print(summary, file=f)

    split = 0

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y_labels[train_index], y_labels[test_index]
        # convert labels to one-hot encodings.
        Y_train      = np_utils.to_categorical(Y_train)
        Y_test       = np_utils.to_categorical(Y_test)
        X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
        
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=17)
        
        checkpointer = ModelCheckpoint(filepath='./Best_chan_20/checkpoint.h5', verbose=1, monitor='val_accuracy', save_best_only=True)
        #checkpointer = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=16, restore_best_weights=True)
        
        # with strategy.scope():

        #     model = EEGNet(nb_classes = 4, Chans = chans, Samples = samples, 
        #             dropoutRate = dr, kernLength = kernLength, F1 = f1, D = d, F2 = f2, 
        #             dropoutType = 'Dropout')
        #     model.compile(
        #         tf.keras.optimizers.Adam(
        #             # Specifies the range of values for the tuner to try
        #             name="Adam"),
        #         loss='categorical_crossentropy',
        #         metrics=['accuracy'])
        
        # fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 400, 
        #                     verbose = 2,
        #                     #validation_data=(X_val, Y_val),
        #                     callbacks=[checkpointer]
        #                     )

           
        #best_model.save(f'best_EEGNet_S{subject}_Split{split}.h5')

        model = tuner.hypermodel.build(best_hyperparameters)

        fittedModel = model.fit(X_train, Y_train, batch_size = 32, epochs = 400, 
                            verbose = 2,
                            validation_data=(X_val, Y_val),
                            callbacks=[checkpointer]
                            )
        best_model = tf.keras.models.load_model('./Best_chan_20/checkpoint.h5')
        
        probs       = best_model.predict(X_test)
        preds       = probs.argmax(axis = -1)  
        acc         = np.mean(preds == Y_test.argmax(axis=-1))
        accuracies.append(acc)

        precision, recall, F1, _ = precision_recall_fscore_support(Y_test.argmax(axis=-1), preds, average='macro')
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(F1)
        
        with open(f'./Best_chan_20/subject{subject}_split{split}.txt', 'w+') as f:
            print('\ntrain index:\n', file=f)
            print(train_index, file=f)
            print('\ntest index:\n', file=f)
            print(test_index, file=f)
            # print('\nBest Params:\n', file=f)
            # print(tuner.get_best_hyperparameters(1)[0], file=f)
            print('\nAccuracy:\n', file=f)
            print(acc, file=f)
            print('\Precision:\n', file=f)
            print(precision, file=f)
            print('\Recall:\n', file=f)
            print(recall, file=f)
            print('\F1:\n', file=f)
            print(F1, file=f)
            print('Using Batch size of test', file=f)

        split += 1
    
    print('Lists')
    print(accuracies)
    print(precisions)
    print(recalls)
    print(F1s)

    av_accuracies[subject] = sum(accuracies)/4
    av_precisions[subject] = sum(precisions)/4
    av_recalls[subject] = sum(recalls)/4
    av_F1s[subject] = sum(F1s)/4

    print("dicts")
    print(av_accuracies)
    print(av_precisions)
    print(av_recalls)
    print(av_F1s)
# -

with open('./Best_chan_20/output_FINAL_RESULTS.txt', 'w+') as f:
    print('\naccuracies:\n', file=f)
    print(av_accuracies, file=f)
    print('\nprecisions:\n', file=f)
    print(av_precisions, file=f)
    print('\nrecalls:\n', file=f)
    print(av_recalls, file=f)
    print('\nF1s:\n', file=f)
    print(av_F1s, file=f)