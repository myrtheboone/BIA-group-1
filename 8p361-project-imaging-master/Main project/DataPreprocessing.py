# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 15:37:27 2022

@author: 20191819
"""
# importing libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
# import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, RandomFlip, RandomRotation
# from tensorflow.keras.layers import Conv2D, MaxPool2D
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.metrics import roc_curve, auc

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

#function to generate training and validation generators for the original data set
def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

      # dataset parameters
      train_path = os.path.join(base_dir, 'train+val', 'train')
      valid_path = os.path.join(base_dir, 'train+val', 'valid')


      RESCALING_FACTOR = 1./255

      # instantiate data generators original data set
      datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)
      train_gen1 = datagen.flow_from_directory(train_path,
                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                               batch_size=train_batch_size,
                                               class_mode='binary',shuffle=False)
      datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)
      val_gen = datagen.flow_from_directory(valid_path,
                                              target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                              batch_size=val_batch_size,
                                              class_mode='binary') 

  
      return train_gen1, val_gen
  
 #function to generate training and validtation generators for one fourth of the data set   
def get_pcam_generators_1_4(base_dir, train_batch_size=9000, val_batch_size=32):
      # dataset parameters
      train_path = os.path.join(base_dir, 'train+val', 'train')
      valid_path = os.path.join(base_dir, 'train+val', 'valid')


      RESCALING_FACTOR = 1./255

      # instantiate data generators original data set
      datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)
      train_gen1 = datagen.flow_from_directory(train_path,
                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                               batch_size=train_batch_size,
                                               class_mode='binary',shuffle=False)

      #create a list to store 1/4 of the images with half label 0 and half label 1
      list_train_gen_1_4 = []
      list_train_gen_1_4.append(train_gen1[0]) #label 0
      list_train_gen_1_4.append(train_gen1[15]) #label 1
      list_train_gen_1_4.append(train_gen1[1]) #label 0 
      list_train_gen_1_4.append(train_gen1[14]) #label 1

      #retrieve a list with the images
      train_gen_1_4_img = []
    
        #retrieve a list with the labels
      train_gen_1_4_lab = []
    
      batch_size = 9000 #the size of the batches
        
      #looping over all the images
      for i in range(4):
        for j in range(batch_size):
           #append the images to the list
           train_gen_1_4_img.append(list_train_gen_1_4[i][0][j])
        
           #append the labels to the list
           train_gen_1_4_lab.append(list_train_gen_1_4[i][1][j])

       #create a data generator function
      datagen = ImageDataGenerator()

      #defining the batch size
      batch_size = 32

      #creating an array that can be put into the flow  function
      train_data_1_4_img = np.array(train_gen_1_4_img, dtype="float")

      #creating the final 1/4 data generator
      train_gen_1_4 = datagen.flow(train_data_1_4_img, train_gen_1_4_lab, batch_size=batch_size,shuffle=True)
      
      datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)
      val_gen = datagen.flow_from_directory(valid_path,
                                              target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                              batch_size=val_batch_size,
                                              class_mode='binary') 
  
      return train_gen_1_4, val_gen
  
def get_pcam_generators_aug(base_dir, train_batch_size=9000, val_batch_size=32):
      # dataset parameters
      train_path = os.path.join(base_dir, 'train+val', 'train')
      valid_path = os.path.join(base_dir, 'train+val', 'valid')


      RESCALING_FACTOR = 1./255 #the rescaling factor

      # instantiate data generators original data set
      datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)
      train_gen1 = datagen.flow_from_directory(train_path,
                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                               batch_size=train_batch_size,
                                               class_mode='binary',shuffle=False)
      
      # instantiate data generators augmented data set
      datagen = ImageDataGenerator(rescale=RESCALING_FACTOR,rotation_range=90,horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
      train_gen2 = datagen.flow_from_directory(train_path,
                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                               batch_size=train_batch_size,
                                               class_mode='binary',shuffle=False)
      # datagen = ImageDataGenerator(rescale=RESCALING_FACTOR,rotation_range=30,horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
      train_gen3 = datagen.flow_from_directory(train_path,
                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                               batch_size=train_batch_size,
                                               class_mode='binary',shuffle=False)

      # datagen = ImageDataGenerator(rescale=RESCALING_FACTOR,rotation_range=30,horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
      train_gen4 = datagen.flow_from_directory(train_path,
                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                               batch_size=train_batch_size,
                                               class_mode='binary',shuffle=False)
      
      datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)
      val_gen = datagen.flow_from_directory(valid_path,
                                              target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                              batch_size=val_batch_size,
                                              class_mode='binary')

      #create a list to store the augmented images
      list_train_gen_aug = []

      #appending one fourth of the original data set
      list_train_gen_aug.append(train_gen1[0])
      list_train_gen_aug.append(train_gen1[15])
      list_train_gen_aug.append(train_gen1[1])
      list_train_gen_aug.append(train_gen1[14])

      #appending one fourth of the augmented data
      list_train_gen_aug.append(train_gen2[0])
      list_train_gen_aug.append(train_gen2[15])
      list_train_gen_aug.append(train_gen2[1])
      list_train_gen_aug.append(train_gen2[14])

      #appending one fourth of the augmented data
      list_train_gen_aug.append(train_gen3[0])
      list_train_gen_aug.append(train_gen3[15])
      list_train_gen_aug.append(train_gen3[1])
      list_train_gen_aug.append(train_gen3[14])

      #appending one fourth of the augmented data
      list_train_gen_aug.append(train_gen4[0])
      list_train_gen_aug.append(train_gen4[15])
      list_train_gen_aug.append(train_gen4[1])
      list_train_gen_aug.append(train_gen4[14])

      #retrieve a list with the images
      train_gen_aug_img = []

      #retrieve a list with the labels
      train_gen_aug_lab = []

      batch_size = 9000 #the size of the batches

      #looping over all the images
      for i in range(16):
          for j in range(batch_size):
              #appending the images to the list
              train_gen_aug_img.append(list_train_gen_aug[i][0][j])
        
              #appending the labels to the list
              train_gen_aug_lab.append(list_train_gen_aug[i][0][j])


      #create a data generator function
      datagen = ImageDataGenerator()

      #creating an array that can be put into the flow  function
      train_data_aug_img = np.array(train_gen_aug_img, dtype="float")

      #defining the batch size
      batch_size = 32

      #creating the final 1/4 data generator
      train_gen_aug = datagen.flow(train_data_aug_img, train_gen_aug_lab, batch_size=batch_size,shuffle=True)
      
  
      return train_gen_aug, val_gen  
  
def visualize_aug(base_dir, train_batch_size=9000, val_batch_size=32):
      # dataset parameters
      train_path = os.path.join(base_dir, 'train+val', 'train')
      valid_path = os.path.join(base_dir, 'train+val', 'valid')


      RESCALING_FACTOR = 1./255 #the rescaling factor

      # instantiate data generators original data set
      datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)
      train_gen1 = datagen.flow_from_directory(train_path,
                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                               batch_size=train_batch_size,
                                               class_mode='binary',shuffle=False)
      
      # instantiate data generators augmented data set
      datagen = ImageDataGenerator(rescale=RESCALING_FACTOR,rotation_range=90,horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
      train_gen2 = datagen.flow_from_directory(train_path,
                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                               batch_size=train_batch_size,
                                               class_mode='binary',shuffle=False)
      # datagen = ImageDataGenerator(rescale=RESCALING_FACTOR,rotation_range=30,horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
      train_gen3 = datagen.flow_from_directory(train_path,
                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                               batch_size=train_batch_size,
                                               class_mode='binary',shuffle=False)

      # datagen = ImageDataGenerator(rescale=RESCALING_FACTOR,rotation_range=30,horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
      train_gen4 = datagen.flow_from_directory(train_path,
                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                               batch_size=train_batch_size,
                                               class_mode='binary',shuffle=False)
      
      datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)
      val_gen = datagen.flow_from_directory(valid_path,
                                              target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                              batch_size=val_batch_size,
                                              class_mode='binary') 
      
      #visualisation of the data augmentation
      iter_size = 4 #how many times the augmentation takes place

      for j in range(iter_size):
          fig, ax = plt.subplots(nrows=1, ncols=iter_size, figsize=(15,15))
          ax[0].axis('off')
          img = train_gen1[j][0][0]
          ax[0].imshow(img)
          ax[0].axis('off')
          img = train_gen2[j][0][0]
          ax[1].imshow(img)
          ax[1].axis('off')
          img = train_gen3[j][0][0]
          ax[2].imshow(img)
          ax[2].axis('off')
          img = train_gen4[j][0][0]
          ax[3].imshow(img)
          ax[3].axis('off')
 
#%% calling the fuction for the original data set

# get the data generators
[train_gen, val_gen] = get_pcam_generators(r'C:\Users\20191819\Documents\school\2021,2022\Q3\ProjectBIA\data')  #Lieke
#[train_gen1, train_gen2, train_gen3, train_gen4] = get_pcam_generators(r'C:\Users\20192823\Documents\3 jaar\Kwartiel 3\BIA', batch_size)  #Jim

#%% calling the function for one fourth of the data set

[train_gen_1_4, val_gen] = get_pcam_generators_1_4(r'C:\Users\20191819\Documents\school\2021,2022\Q3\ProjectBIA\data')

#%% calling the function for the augmented data set

[train_gen_aug, val_gen] = get_pcam_generators_aug(r'C:\Users\20191819\Documents\school\2021,2022\Q3\ProjectBIA\data')

#%% creating the 1/4 data set

#create a list to store 1/4 of the images with half label 0 and half label 1
list_train_gen_1_4 = []
list_train_gen_1_4.append(train_gen1[0]) #label 0
list_train_gen_1_4.append(train_gen1[15]) #label 1
list_train_gen_1_4.append(train_gen1[1]) #label 0 
list_train_gen_1_4.append(train_gen1[14]) #label 1

#retrieve a list with the images
train_gen_1_4_img = []

#retrieve a list with the labels
train_gen_1_4_lab = []

batch_size = 9000 #the size of the batches

#looping over all the images
for i in range(4):
    for j in range(batch_size):
        #append the images to the list
        train_gen_1_4_img.append(list_train_gen_1_4[i][0][j])

        #append the labels to the list
        train_gen_1_4_lab.append(list_train_gen_1_4[i][1][j])

#create a data generator function
datagen = ImageDataGenerator()

#defining the batch size
batch_size = 32

#creating an array that can be put into the flow  function
train_data_1_4_img = np.array(train_gen_1_4_img, dtype="float")

#creating the final 1/4 data generator
train_gen_1_4 = datagen.flow(train_data_1_4_img, train_gen_1_4_lab, batch_size=batch_size,shuffle=True)

#%% Visualizing the data augmentation (4 example plots)

#visualisation of the data augmentation
iter_size = 4 #how many times the augmentation takes place

for j in range(iter_size):
    fig, ax = plt.subplots(nrows=1, ncols=iter_size, figsize=(15,15))
    ax[0].axis('off')
    img = train_gen1[j][0][0]
    ax[0].imshow(img)
    ax[0].axis('off')
    img = train_gen2[j][0][0]
    ax[1].imshow(img)
    ax[1].axis('off')
    img = train_gen3[j][0][0]
    ax[2].imshow(img)
    ax[2].axis('off')
    img = train_gen4[j][0][0]
    ax[3].imshow(img)
    ax[3].axis('off')

#%% creating the augmented data set

#create a list to store the augmented images
list_train_gen_aug = []

#appending one fourth of the original data set
list_train_gen_aug.append(train_gen1[0])
list_train_gen_aug.append(train_gen1[15])
list_train_gen_aug.append(train_gen1[1])
list_train_gen_aug.append(train_gen1[14])

#appending one fourth of the augmented data
list_train_gen_aug.append(train_gen2[0])
list_train_gen_aug.append(train_gen2[15])
list_train_gen_aug.append(train_gen2[1])
list_train_gen_aug.append(train_gen2[14])

#appending one fourth of the augmented data
list_train_gen_aug.append(train_gen3[0])
list_train_gen_aug.append(train_gen3[15])
list_train_gen_aug.append(train_gen3[1])
list_train_gen_aug.append(train_gen3[14])

#appending one fourth of the augmented data
list_train_gen_aug.append(train_gen4[0])
list_train_gen_aug.append(train_gen4[15])
list_train_gen_aug.append(train_gen4[1])
list_train_gen_aug.append(train_gen4[14])

#retrieve a list with the images
train_gen_aug_img = []

#retrieve a list with the labels
train_gen_aug_lab = []

batch_size = 9000 #the size of the batches

#looping over all the images
for i in range(16):
    for j in range(batch_size):
        #appending the images to the list
        train_gen_aug_img.append(list_train_gen_aug[i][0][j])
        
        #appending the labels to the list
        train_gen_aug_lab.append(list_train_gen_aug[i][0][j])


#create a data generator function
datagen = ImageDataGenerator()

#creating an array that can be put into the flow  function
train_data_aug_img = np.array(train_gen_aug_img, dtype="float")

#defining the batch size
batch_size = 32

#creating the final 1/4 data generator
train_gen_aug = datagen.flow(train_data_aug_img, train_gen_aug_lab, batch_size=batch_size,shuffle=True)