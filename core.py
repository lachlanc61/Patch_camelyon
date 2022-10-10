
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from keras import Sequential 

import src.utils as utils
import src.tfmodel as tfmodel
import src.vis as vis

"""
Practice project - histology

"""
#-----------------------------------
#Vars
#-----------------------------------

CONFIG_FILE='config.yaml'
DSETNAME='patch_camelyon'
#-----------------------------------
#INITIALISE
#-----------------------------------

config=utils.readcfg(CONFIG_FILE)

wdirname=config['wdirname']
odirname=config['odirname']
batchlen=config['batchlen']

spath, wdir, odir, checkpoint_path, checkpoint_dir = utils.initialise(config)

#-----------------------------------
#MAIN START
#-----------------------------------

dtrain, dval, dtest, dsinfo = utils.datainit(config, DSETNAME)

timg, tlabels, vimg, vlabels, testimg, testlabels = utils.batchcheck(dtrain, dval, dtest, batchlen)


if config['PREPLOT']:
    vis.preplot(config)

#we are overfitting pretty heavily, try regularisation
l2reg=tf.keras.regularizers.L2(config['lamda'])
#kernel_regularizer=l2reg
  #not great, slow and not a big improvement, leave it out for now


model = tfmodel.build(config)

model, fitlog = tfmodel.train(model, dtrain, dval, config, checkpoint_path)

vis.layerplot(config, model, timg, tlabels, odir)

print("CLEAN EXIT")
exit()

"""
performance log

3 layers 256 128, batch 512 - 0.6 vacc
4 layers 256 128, batch 512 - 0.63 vacc
4 layers 256 128 batch 8k  - ~0.65 vacc - v slow

4+1 batch 512               -   ~0.7 vacc
  1 conv2D 32,3             - seems unstable, vacc varying 0.65-0.75
  4 dense 256,128,56,12 
    4250 ms/step on CPU
    ~27 ms/step on GPU

try L2_regularisation on all layers
  - much slower (1.2 sec), still 0.7-0.75 vacc
  - maybe more stable but seems not worth

    Ok here's an example
    https://medium.com/analytics-vidhya/deep-learning-tutorial-patch-camelyon-data-set-d0da9034550e
      6 conv2D layers + 6 "maxpool" layers - what is this?
      3 dense layers
      gets ~0.75, 150ms/step

try adding this maxpool layer  - 0.72 vacc, more stable?
  faster - 17ms/step


up to three conv+maxpool layers
  256 1024 256 
  touches 0.77 but very slow to train





"""

"""
resources

really good explanation of maxpooling here
https://www.youtube.com/watch?v=ZjM_XQa5s6s

also nice CNN overview
https://www.youtube.com/watch?v=YRhxdVk_sIs

"""