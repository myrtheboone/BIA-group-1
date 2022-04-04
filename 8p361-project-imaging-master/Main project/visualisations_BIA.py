# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 12:12:30 2022

@author: 20192024
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 12:12:30 2022

@author: 20192024
"""
# =============================================================================
# VISUALISATION FUNCTIONS
# =============================================================================

# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, auc
import numpy as np
import keras
from sklearn.metrics import roc_curve, auc
from DataPreprocessing import get_pcam_generators
# Plot roc curve with auc score in plot

def plot_roc_curve(fpr,tpr, auc_score, title): 
  """Function that takes as input fpr and tpr of model 
  and optional argument the title of the plot"""  
  plt.plot(fpr,tpr, label="ROC curve (area = {0:0.2f})".format(auc_score)) 
  plt.axis([0,1,0,1]) 
  plt.xlabel('False Positive Rate') 
  plt.ylabel('True Positive Rate') 
  plt.title(title)
  plt.legend(loc='lower right')
  plt.show()    
  


# Plot accuracy and loss curves from model history

def accuracy_loss_curves(history, epochs):
    """Function that plots the loss and accuracy curves 
    after training process based on model history"""

    # summarize history for accuracy
    range_epochs = np.arange(1,epochs+1,1)
    
    plt.ylim([0,1])
    plt.plot(range_epochs, history.history['accuracy'])
    plt.plot(range_epochs, history.history['val_accuracy'])
    plt.title('AlexNet (DropOut 0.2, SGD) accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(range_epochs, history.history['loss'])
    plt.plot(range_epochs, history.history['val_loss'])
    plt.title('AlexNet (DropOut 0.2, SGD) loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def heatmap_confusion(model, y_true, y_prob):
    """Function that plots the confusion matrix for a specified model with a test set containing the true classes and the predicted probabilities
    Output is a confusion matrix in the form of a seaborn heatmap. Predicted is on the x-axis and the true values are on the y-axis"""
    cm = confusion_matrix(y_true, y_prob>0.5)
    labels = [0,1]
    class_names = labels
    
    # Plot confusion matrix in a beautiful manner
    fig = plt.figure(figsize=(16, 14))
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
    sns.set(font_scale=2.0) # Adjust to fit
    ax.set_xlabel('Predicted', fontsize=35)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(class_names, fontsize = 30)
    ax.xaxis.tick_bottom()
    
    ax.set_ylabel('True', fontsize=35)
    ax.yaxis.set_ticklabels(class_names, fontsize = 30)
    plt.yticks(rotation=0)
    
    plt.title('Confusion matrix - healthy (0) or metastasis (1)', fontsize=40)
    plt.show()

