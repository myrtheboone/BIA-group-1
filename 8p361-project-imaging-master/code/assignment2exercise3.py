# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:26:44 2022

@author: 20191819
"""
#%% importing libraries

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf


# import required packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

#%%loading the dataset and preprocessing the data

# load the dataset using the builtin Keras method
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# derive a validation set from the training set
# the original training set is split into 
# new training set (90%) and a validation set (10%)
X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)



# the shape of the data matrix is NxHxW, where
# N is the number of images,
# H and W are the height and width of the images
# keras expect the data to have shape NxHxWxC, where
# C is the channel dimension
X_train = np.reshape(X_train, (-1,28,28,1)) 
X_val = np.reshape(X_val, (-1,28,28,1))
X_test = np.reshape(X_test, (-1,28,28,1))


# convert the datatype to float32
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')


# normalize our data values to the range [0,1]
X_train /= 255
X_val /= 255
X_test /= 255


# convert 1D class arrays to 10D class matrices
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
y_test = to_categorical(y_test, 10)
#%% converting the 10D class matrices to 4D class matrices

#creating memory to save the new classes
y_train_4 = np.zeros((54000,4))
y_val_4 = np.zeros((6000,4))
y_test_4 = np.zeros((10000,4))

#a for loop over the indices of y_train
for i in range(y_train.shape[0]):
    
    ytr = y_train[i]
    
    #filling the new classes for the training set
    if ytr[1] ==1 or ytr[7]==1:
        y_train_4[i][0] = 1
    if ytr[0] ==1 or ytr[6]==1 or ytr[8]==1 or ytr[9]==1:
        y_train_4[i][1] =1
    if ytr[2]==1 or ytr[5]==1:
        y_train_4[i][2]=1
    if ytr[3]==1 or ytr[4]==1:
        y_train_4[i][3]=1

#a for loop over the indices of y_val
for i in range(y_val.shape[0]):
    
    yva= y_val[i]

    #filling in the new classes for the validation set
    if yva[1] ==1 or yva[7]==1:
        y_val_4[i][0] = 1
    if yva[0] ==1 or yva[6]==1 or yva[8]==1 or yva[9]==1:
        y_val_4[i][1] =1
    if yva[2]==1 or yva[5]==1:
        y_val_4[i][2]=1
    if yva[3]==1 or yva[4]==1:
        y_val_4[i][3]=1    
        
#a for loop over the indices of y_test
for i in range(y_test.shape[0]):
    
    yte = y_test[i]

    #filling in the new classes for the test set
    if yte[1] ==1 or yte[7]==1:
        y_test_4[i][0] = 1
    if yte[0] ==1 or yte[6]==1 or yte[8]==1 or yte[9]==1:
        y_test_4[i][1] =1
    if yte[2]==1 or yte[5]==1:
        y_test_4[i][2]=1
    if yte[3]==1 or yte[4]==1:
        y_test_4[i][3]=1        


#%% Creating the model for exercise 3
    
    #model_3: Two hidden layers of 64 neurons

model = Sequential()
# flatten the 28x28x1 pixel input images to a row of pixels (a 1D-array)
model.add(Flatten(input_shape=(28,28,1))) 
# fully connected layer with 64 neurons and ReLU nonlinearity
model.add(Dense(64, activation='relu'))
# fully connected layer with 64 neurons and ReLU nonlinearity
model.add(Dense(64, activation='relu'))
# output layer with 10 nodes (one for each class) and softmax nonlinearity
model.add(Dense(4, activation='softmax')) 

# compile the model_3
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# use this variable to name your model
model_name="model_3"

# create a way to monitor our model in Tensorboard
tensorboard = TensorBoard("logs/" + model_name)

# train the model
model.fit(X_train, y_train_4, batch_size=32, epochs=20, verbose=1, validation_data=(X_val, y_val_4), callbacks=[tensorboard])

#calculate the scores of the model
score = model.evaluate(X_test, y_test_4, verbose=0)

#print the scores
print("Loss: ",score[0])
print("Accuracy: ",score[1])
