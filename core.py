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
#CLASSES
#-----------------------------------

#-----------------------------------
#INITIALISE
#-----------------------------------

starttime = time.time()             #init timer
print(sys.version)
gpu_available = (len(tf.config.list_physical_devices('GPU')) > 0)

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
dvalid = data['validation']
dtest = data['test']

timg, tlabels = next(iter(dtrain))
vimg, vlabels = next(iter(dvalid))

#Checking the label shape
print("labelshape",vlabels.shape)

#Checking the image shape
print("imgshape",timg.shape)


fig, ax = plt.subplots(1,3, figsize=(12,4))
fig.tight_layout(pad=0.1)

for i in np.arange(3):
    ax[i].imshow(timg[:,:,i])

plt.show()
#open the data 

print("CLEAN EXIT")
exit()
