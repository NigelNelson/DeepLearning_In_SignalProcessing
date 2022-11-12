# Used to perform a hyper parameter search for optimal values

# Uncomment below if unable to import "keras_tuner"
# %pip install keras-tuner --upgrade
import keras_tuner as kt
import tensorflow as tf
import pandas as pd
# from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib
import time
####################################################################
# Inner Voice Imports
import os
import mne 
import pickle
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt

from Inner_Speech_Dataset.Python_Processing.Data_extractions import  Extract_data_from_subject, Extract_data_multisubject
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

####################################################################
# Used to verify all expected GPUs are available
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print('GPUS:')
print(get_available_gpus())
print()
""
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
N_S = 1   #[1 to 10]
subjects = [2,3,4,5,6,7,8]

""
### Data Set-up

print('Extracting single subject')
# Load all trials for a sigle subject
X, Y = Extract_data_from_subject(root_dir, N_S, datatype)
#X, Y = Extract_data_multisubject(root_dir, subjects, datatype)

""
# Cut usefull time. i.e action interval
X = Select_time_window(X = X, t_start = t_start, t_end = t_end, fs = fs)
X, Y = Filter_by_condition(X, Y, "INNER")
y_labels = Y[:,1]

#X_train, X_test, Y_train, Y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)

# left_channels = np.array([0,
#                         1,
#                         3,
#                         4,
#                         7,
#                         8,
#                         9,
#                         14,
#                         18,
#                         19,
#                         21,
#                         22,
#                         23,
#                         81,
#                         83,
#                         90,
#                         94,
#                         95,
#                         100,
#                         101,
#                         105,
#                         110,
#                         114,
#                         116,
#                         121,
#                         125
#                         ])
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

print('\n\n\n\n\n')
print(best_channel_mask)
print(X.shape)


x_min = X.min(axis=(2), keepdims=True)
x_max = X.max(axis=(2), keepdims=True)

x_min.shape

X = (X - x_min)/(x_max-x_min)

with open('output_left.txt', 'w+') as f:
    print(f'Shape of X is: {X.shape}', file=f)


kernels, chans, samples = 1, X.shape[1], X.shape[2]

# convert labels to one-hot encodings.
Y_train      = np_utils.to_categorical(y_labels)
#Y_test       = np_utils.to_categorical(Y_test)
X_train      = X.reshape(X.shape[0], chans, samples, kernels)
#X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)



""
def build_model(hp):  
    """
    Method used by the tuner to build models
    param: hp: the hyperparameters
    """
    # Create the keras model


    dropoutRate = hp.Float(
                    "dropout",
                    0.35,
                    0.75,
                    step=0.05
                )

    kernLength = hp.Int(
                    "kernel_length",
                    30,
                    120,
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
                    30,
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
               dropoutRate = dropoutRate, kernLength = kernLength, 
               F1 = temporal_filters, D = spatial_filters, F2 = pointwise_filters, 
               dropoutType = 'Dropout')

    
    
    # Convert model to run on multiple GPUs
    #parallel_model = multi_gpu_model(model, gpus=8)
    
    model.compile(
        tf.keras.optimizers.Adam(
            # Specifies the range of values for the tuner to try
            learning_rate= hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log"),
            name="Adam"),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=300,
    hyperband_iterations=5,
    overwrite=True,
    distribution_strategy=strategy,
    directory="search_results",
    project_name='EEGNet_tuning3')

tuner.search(X_train,
             Y_train,
            validation_split=0.2,
            epochs=300,
            verbose=2)

# Save the best model from the hyperparameter search
best_model = tuner.get_best_models(1)[0]
best_model.save('best_channels_EEGNet.h5')

# Print the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

# Print the results of the hyperparameter search
summary = tuner.results_summary()


with open('output_best_channels.txt', 'w+') as f:
    print(f'Best dropout: {best_hyperparameters.get("dropout")}', file=f)
    print(f'Best kernel_length: {best_hyperparameters.get("kernel_length")}', file=f)
    print(f'Best temporal_filters: {best_hyperparameters.get("temporal_filters")}', file=f)
    print(f'Best pointwise_filters: {best_hyperparameters.get("pointwise_filters")}', file=f)
    print(f'Best spatial_filters: {best_hyperparameters.get("spatial_filters")}', file=f)
    print(f'Best lr: {best_hyperparameters.get("lr")}', file=f)
    print(summary, file=f)

model = tuner.hypermodel.build(best_hyperparameters)
history = model.fit(X_train, Y_train, epochs=300, validation_split=0.2)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

with open('output_best_channels_best_epoch.txt', 'w+') as f:
    print(f'Best epoch: {best_epoch}', file=f)
