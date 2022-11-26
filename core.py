
import numpy as np
import matplotlib.pyplot as plt

import src.utils as utils
import src.tfmodel as tfmodel
import src.vis as vis

"""
CNN classifier for histology

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

#initialise the data into train, validation & test sets, and extract dataset info
#   normalisation and augmentation applied here
dtrain, dval, dtest, dsinfo = utils.datainit(config, DSETNAME)

#extract a single image and print the batch size
timg, tlabels, vimg, vlabels, testimg, testlabels = utils.batchcheck(dtrain, dval, dtest, batchsize)

#produce an initial plot of 12 random images, if requested
if config['PREPLOT']:
    vis.preplot(config, timg, tlabels)

#build the model
#   flags in config can be used to set hyperparameters
model = tfmodel.build(config)

#train the model, producing a log of the training process
model, fitlog = tfmodel.train(model, dtrain, dval, config, checkpoint_path, odir)

#produce a plot from a single random image, visualising a subset of the filters
vis.layerplot(config, model, timg, tlabels, odir)

print("CLEAN EXIT")
exit()