# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:57:39 2022

@author: 20191974
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

import numpy as np
from visualisations_BIA import *



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D, Conv2D,BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.metrics import roc_curve, auc

#Import libraries necessary for hyperparameter tuning

from keras_tuner import HyperParameters
import keras_tuner as kt


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
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary', shuffle = False) 
     
    

     return train_gen, val_gen

train_gen, val_gen = get_pcam_generators('C:/Users//20192024//Documents//Project_BIA')




#Instantiation
AlexNet = Sequential()

#1st Convolutional Layer
AlexNet.add(Conv2D(filters=96, activation = 'relu', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3), kernel_size=(11,11), strides=(4,4), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#2nd Convolutional Layer
AlexNet.add(Conv2D(filters=256, activation = 'relu', kernel_size=(5, 5), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#3rd Convolutional Layer
AlexNet.add(Conv2D(filters=384, activation = 'relu', kernel_size=(3,3), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())

#4th Convolutional Layer
AlexNet.add(Conv2D(filters=384, activation = 'relu', kernel_size=(3,3), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())

#5th Convolutional Layer
AlexNet.add(Conv2D(filters=256, activation = 'relu', kernel_size=(3,3), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#Passing it to a Fully Connected layer
AlexNet.add(Flatten())
# 1st Fully Connected Layer
AlexNet.add(Dense(4096, activation = 'relu', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
AlexNet.add(BatchNormalization())
# Add Dropout to prevent overfitting
AlexNet.add(Dropout(0.2))

#2nd Fully Connected Layer
AlexNet.add(Dense(4096, activation = 'relu'))
AlexNet.add(BatchNormalization())
#Add Dropout
AlexNet.add(Dropout(0.2))

#3rd Fully Connected Layer
AlexNet.add(Dense(1000, activation = 'relu'))
AlexNet.add(BatchNormalization())
#Add Dropout
AlexNet.add(Dropout(0.2))

#Output Layer
AlexNet.add(Dense(1, activation = 'sigmoid'))
opt = SGD(learning_rate=0.01, momentum=0.95) #Adam already uses some type of momentum so we do not have to specify it 
AlexNet.compile(opt, loss = 'binary_crossentropy', metrics=['accuracy'])



model_name = 'model__alexnet_SGD_clownings' 
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_alexnet_json = AlexNet.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_alexnet_json)


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]

# train the model

train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size




history = AlexNet.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=30, callbacks=callbacks_list)


# Plotting the accuracy and loss curves of validation and training set

accuracy_loss_curves(history, 30)


# Predict on validation set with AlexNet

val_prob = AlexNet.predict(val_gen)
filenames=val_gen.filenames
val_true_labels = []

for i in filenames:
    val_true_labels.append(int(i[0]))

val_true_array = np.array(val_true_labels)

val_prob_array = np.array(val_prob)

val_true_array = val_true_array.reshape(16000,1)

# Plotting the ROC curve

fpr , tpr , thresholds = roc_curve(val_true_labels, val_prob)
auc_score = auc(fpr, tpr)
plot_roc_curve(fpr,tpr, auc_score, 'ROC curve - AlexNet (DropOut 0.2, SGD)') 

# Plotting the confusion matrix

heatmap_confusion(AlexNet, val_true_array, val_prob_array)

# Let the model predict on the test set

test_path = os.path.join('C:/Users//20192024//Documents//Project_BIA', 'test', 'test')
RESCALING_FACTOR = 1./255

# instantiate data generators
datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

test_gen = datagen.flow_from_directory(test_path,
                                        target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                        class_mode='binary')
test_prob = AlexNet.predict(test_gen)