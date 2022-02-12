# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 09:32:43 2022

@author: 20192024
"""
import imageio
import matplotlib.pyplot as plt


# Displaying two classes : healthy tissue and tissue with metastasis

path_healthy = '.\\train+val\\train\\0'


path_metastasis = '.\\train+val\\train\\1'

train1_healthy = imageio.imread(path_healthy + '\\00a2f36418691726cf9fe87bc6c87f2c17b948ae.jpg')
train2_healthy = imageio.imread(path_healthy+'\\00eaa2f7083ad26570fd178e87973d39a7581182.jpg')
train3_meta = imageio.imread(path_metastasis+'\\00bcb0f6a63ca0de2d88895137dfd3c7c721b402.jpg')
train4_meta = imageio.imread(path_metastasis+'\\0a7fe123f688fcf7a81fa4490c92cc6905f77fe1.jpg')


fig, axs = plt.subplots(2,2)
fig.suptitle('Four images')
axs[0,0].imshow(train1_healthy)
axs[0,1].imshow(train2_healthy)
axs[1,0].imshow(train3_meta)
axs[1,1].imshow(train4_meta)
axs[0,0].title.set_text('Healthy')
axs[0,1].title.set_text('Healthy')
axs[1,0].title.set_text('Metastasis')
axs[1,1].title.set_text('Metastasis')


