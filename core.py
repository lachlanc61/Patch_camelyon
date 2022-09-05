import numpy as np
import matplotlib.pyplot as plt
import os,datetime, time, gc
import sys

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

from keras import Sequential 
from keras import layers
from keras import optimizers
from keras import regularizers

import config
import src.utils as utils


"""
Project description here

"""
#-----------------------------------
#VARIABLES
#-----------------------------------

batchlen = 128
buffer=5000

#-----------------------------------
#FUNCTIONS
#-----------------------------------

def normalise(img, labels):
  """
  simple normaliser to pass to tfds.map
    scales 0-255 to 0-1
    receives tensor tuple, returns same 
  """
  img = tf.cast(img, tf.float32)
  img = img/255.
  return img, labels

#-----------------------------------
#INITIALISE
#-----------------------------------

starttime = time.time()             #init timer
print(f'python version: {sys.version}')
print(f'tensorflow version: {tf.__version__}')

gpu_available = (len(tf.config.list_physical_devices('GPU')) > 0)
print(f'GPU: {gpu_available}')
#-----------------------------------
#MAIN START
#-----------------------------------
"""
weird TF bug - datasets going in literal ~ directory rather than home
set TDFS_DATA_DIR in bashrc - see if this helps next time
"""

print("---------------")
data, info = tfds.load('patch_camelyon', with_info = True, as_supervised = True)

dtrain = data['train']
dvalidation = data['validation']
dtest = data['test']

#normalise all to range(0,1) - speeds up ML calc
# tfds.map performs (function) on each element of the array
dtrain = dtrain.map(normalise)
dvalidation = dvalidation.map(normalise)
dtest = dtest.map(normalise)

#shuffle the training data to use a different set each time
dtrain=dtrain.shuffle(buffer)

#get a sub-batch for training
dtrain = dtrain.batch(batchlen) 
dvalidation = dvalidation.batch(batchlen) 
dtest = dtest.batch(batchlen) 

#extract tensor elements
#   iter converts to iterable object
#   next extracts element from each
#   comes as (image, label) tuple

timg, tlabels = next(iter(dtrain))
vimg, vlabels = next(iter(dvalidation))

#check shapes for both
#should correspond to batch size
print("labelshape:",vlabels.shape)
print("imgshape:",timg.shape)
print("batch size:",batchlen)

#plot 12 random images as RGB, including label as true/false 
fig, ax = plt.subplots(2,6, figsize=(12,5))
fig.tight_layout(pad=0.1)

for i,ax in enumerate(ax.flat):
    rand = np.random.randint(batchlen)    
    ax.imshow(timg[rand,:,:,:])
    ax.set_title(bool(tlabels.numpy()[rand]))
    ax.set_axis_off()

plt.show()
#open the data 

print("CLEAN EXIT")
exit()
