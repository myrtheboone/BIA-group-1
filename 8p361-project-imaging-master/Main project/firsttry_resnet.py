# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:33:15 2022

@author: 20192024
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

import numpy as np



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
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



base_model = Sequential()
base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
base_model.add(Dense(1, activation='sigmoid'))
# add dropout layer here

base_model.compile(optimizer = RMSprop(lr=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])


# save the model and weights
# JENS LET OP HIERONDER AANPASSEN

model_name = 'model_dropout_0.5' #andere keer 0.75 (voor Myrthe)
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_dropout_05_json = base_model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_dropout_05_json)


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]

# train the model

train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size




history = base_model.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=1, callbacks=callbacks_list)


#Plot ROC curves of ResNet

val_prob = base_model.predict(val_gen)
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

def plot_roc_curve(fpr,tpr): 
  plt.plot(fpr,tpr, label="ROC curve (area = {0:0.2f})".format(auc_score)) 
  plt.axis([0,1,0,1]) 
  plt.xlabel('False Positive Rate') 
  plt.ylabel('True Positive Rate') 
  plt.title('ROC curve - model with dense layers')
  plt.legend(loc='lower right')
  plt.show()    
  
plot_roc_curve (fpr,tpr) 



# Plot accuracy and loss curves of ResNet from model history

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Save model

base_model.save('C:/Users//20192024//Documents//Project_BIA//BIA-group-1//8p361-project-imaging-master//Main project')