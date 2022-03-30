# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:19:50 2022

@author: 20192024
"""
import tensorflow as tf
from tensorflow import keras

from kerastuner.applications import HyperResNet
from kerastuner.tuners import Hyperband

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

import numpy as np



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop

from sklearn.metrics import roc_curve, auc


IMAGE_SIZE = 96


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     train_path = os.path.join(base_dir, 'train+val', 'train')
     valid_path = os.path.join(base_dir, 'train+val', 'valid')


     RESCALING_FACTOR = 1./255

     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary', shuffle = False)

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary', shuffle = False) 

     return train_gen, val_gen

train_gen, val_gen = get_pcam_generators('C:/Users//20192024//Documents//Project_BIA')

# We tune parameters for the version of the ResNet,
# the depth of the convolutional layer blocks, the learning rate
# and the optimization algorithm.

hypermodel = HyperResNet(input_shape=(96,96,3), classes=2)

from kerastuner import HyperParameters
hp = HyperParameters()
hp.Choice('learning_rate', values=[1e-3, 1e-4])
hp.Fixed('optimizer', value='adam')

# With tune_new_entries = False, the tuner will use the tuning configuration 
# from the HyperResNet source code when it needs a parameter 
# that is not defined in my HyperParameters object.

tuner = Hyperband(
    hypermodel,
    objective='val_accuracy',
    hyperparameters=hp,
    tune_new_entries=False,
    max_trials=20)

tuner.search(train_gen,
             validation_data=val_gen,
             epochs=10,
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])


best_model = tuner.get_best_models(1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]



train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hyperparameters)
history = model.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=10)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hyperparameters)

# Save the model
model_name = 'model_hyperparamtuning_myrthe' 
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_hyperparamtuning_json = hypermodel.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_hyperparamtuning_json)


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]

# Retrain the model
hypermodel.fit(train_gen, steps_per_epoch = train_steps, 
               validation_data=val_gen, validation_steps=val_steps, epochs=best_epoch)


