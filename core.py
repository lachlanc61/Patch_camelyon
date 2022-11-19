
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
batchsize=config['batchsize']

spath, wdir, odir, checkpoint_path, checkpoint_dir = utils.initialise(config)

#-----------------------------------
#MAIN START
#-----------------------------------

dtrain, dval, dtest, dsinfo = utils.datainit(config, DSETNAME)

timg, tlabels, vimg, vlabels, testimg, testlabels = utils.batchcheck(dtrain, dval, dtest, batchsize)


if config['PREPLOT']:
    vis.preplot(config, timg, tlabels)

#we are overfitting pretty heavily, try regularisation
l2reg=tf.keras.regularizers.L2(config['lamda'])
#kernel_regularizer=l2reg
  #not great, slow and not a big improvement, leave it out for now


model = tfmodel.build(config)

model, fitlog = tfmodel.train(model, dtrain, dval, config, checkpoint_path)

vis.layerplot(config, model, timg, tlabels, odir)

print("CLEAN EXIT")
exit()