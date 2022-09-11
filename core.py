import numpy as np
import matplotlib.pyplot as plt
import sys 
import os
import gc
import time 

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

from keras import Sequential 


"""
Practice project - histology

"""
#-----------------------------------
#VARIABLES
#-----------------------------------
DOTRAIN=False       #train model from scratch
PREPLOT=False       #plot some images before running model
POSTPLOT=True       #plot accuracy and loss over time, only used if training
LAYEROUTPLOT=True  #plot layer outputs - not working yet



#workdir and inputfile
# NB: both directories in gitignore - would need to create locally in clone
wdirname='train'     #working directory relative to script    
odirname='out'      #output directory relative to script

#cptoload='/home/lachlan/CODEBASE/Patch_camelyon/train/220910_vacc77/cp-0044.ckpt'
cptoload='/home/lachlan/CODEBASE/Patch_camelyon/train/220911_vacc75_c2d64/cp-0032.ckpt'

             #location of checkpoint to pre-load 
             #manually set for now

batchlen = 256    #size of batch for fitting
buffer=5000       #buffer for shuffling
nepochs=100       #no. epochs
stoplim=25        #patience for early stop
batch_size=32     #size of batch for training
lamda=0.0001      #regularisation param
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

#if we're not training, hide GPU to avoid it going OOM when viewing layer outputs
#   sure there must be a way to do this more cleanly
if not DOTRAIN:
  tf.config.set_visible_devices([], 'GPU')

print(f'python version: {sys.version}')
print(f'tensorflow version: {tf.__version__}')

gpu_available = (len(tf.config.list_physical_devices('GPU')) > 0)
print(f'GPU: {gpu_available}')


#initialise directories relative to script
script = os.path.realpath(__file__) #_file = current script
spath=os.path.dirname(script) 
wdir=os.path.join(spath,wdirname)
odir=os.path.join(spath,odirname)

checkpoint_path = os.path.join(wdir,"cp-{epoch:04d}.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)

print("script path:", spath)
print("working path:", wdir)
print("output path:", odir)
print("cp:", checkpoint_path)
print("cp:", checkpoint_dir)
print("---------------")


#-----------------------------------
#MAIN START
#-----------------------------------

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
testimg, testlabels  = next(iter(dtest))

#check shapes for both
#should correspond to batch size
print("labelshape:",vlabels.shape)
print("imgshape:",timg.shape)
print("batch size:",batchlen)
if PREPLOT:
  #plot 12 random images as RGB, including label as true/false 
  fig, ax = plt.subplots(2,6, figsize=(12,5))
  fig.tight_layout(pad=0.1)

  for i,ax in enumerate(ax.flat):
      rand = np.random.randint(batchlen)    
      ax.imshow(timg[rand,:,:,:])
      ax.set_title(bool(tlabels.numpy()[rand]))
      ax.set_axis_off()

  plt.show()
  exit()

#we are overfitting pretty heavily, try regularisation
l2reg=tf.keras.regularizers.L2(lamda)
#kernel_regularizer=l2reg
  #not great, slow and not a big improvement, leave it out for now


"""
MODEL HERE

  convolution layer (Nfilters,Npx,...)
    basically a set of N filters acting on a window of MxM px slid across whole image
      output shape x,y,Nfilters
  
  max pooling layer (Npx)
    sliding Npx x Npx window
    outputs max value within window
      effectively downsamples image retaining max
      https://www.youtube.com/watch?v=ZjM_XQa5s6s

  use relu function instead of signmoid for all but output layer
  basically max(0,val)
  = passthrough if above 0
    more responsive across entire range compared to sigmoid

"""

#Initialise basic TF model

model = Sequential(
    [
        keras.layers.Conv2D(64,3, padding='same', activation='relu', input_shape=[96, 96, 3], name="block1_conv1"),
        keras.layers.MaxPooling2D(2, name="block1_pool"),
        keras.layers.Conv2D(32,3, padding='same', activation='relu', name="block2_conv1"),
        keras.layers.MaxPooling2D(2, name="block2_pool"),
        keras.layers.Conv2D(16,3, padding='same', activation='relu', name="block3_conv1"),
        keras.layers.MaxPooling2D(2, name="block3_pool"),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu', name="dense1"),
        keras.layers.Dense(128, activation='relu', name="dense2"),
        keras.layers.Dense(56, activation='relu', name="dense3"),
        keras.layers.Dense(12, activation='relu', name="dense4"), 
        keras.layers.Dense(1, activation='sigmoid', name="predictions") 
    ], name = "my_model" 
)   


"""
model = Sequential(
    [
        keras.layers.Conv2D(256,3, padding='same', activation='relu', input_shape=[96, 96, 3]),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(1024,3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256,3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(56, activation='relu'),
        keras.layers.Dense(12, activation='relu'), 
        keras.layers.Dense(1, activation='sigmoid') 
    ], name = "my_model" 
)   
"""


#view model summary
model.summary()

#compile the model
#   acc = track accuracy vs train/val sets
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.0005),
    metrics=['acc'],
)


#Callbacks
#   (special utilities executed during training)
# https://blog.paperspace.com/tensorflow-callbacks/

#stop refinement early if val stops improving
#https://towardsdatascience.com/a-practical-introduction-to-early-stopping-in-machine-learning-550ac88bc8fd
stopcond=keras.callbacks.EarlyStopping(monitor='val_loss', patience=stoplim)


#save model during training every 5 batches
# save best model only based on val_loss

#ok i am stumped on saving best model
#   get warning "Can save best model only with val_acc available, skipping"
#   looks like val_acc not making it into model.metrics somehow

#BUT val_loss works for early stoping callback. both are available in history
#I guess just save every 5 for now.... creates a pile of checkpoints, 
#       will miss very best one unless save every epoch

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                               #  save_best_only=True,
                                                 monitor='val_acc',
                                                 verbose=1,
                                                 save_freq=1*batch_size)


#TRAIN/LOAD

#if training, do it
#else load model from checkpoint variable (manually set)

if DOTRAIN:
  #run the fit, saving params to fitlog 
  #   epochs = N. cycles
  #   validation data is tested each epoch
  fitlog = model.fit(
      timg,tlabels,
      epochs=nepochs,
      validation_data = (vimg, vlabels),
      validation_freq = 1,
      callbacks=[stopcond,cp_callback],
      batch_size=batch_size,
      verbose=1
  )

  print(model.metrics_names)

  #demonstrating val_acc is present
  for key in fitlog.history:
    print(key)

  #extract metrics for plotting
  tacc = fitlog.history['acc']
  tloss = fitlog.history['loss']
  vacc = fitlog.history['val_acc']
  vloss = fitlog.history['val_loss']


  #plot fit progress against train and validation data
  if POSTPLOT:
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    fig.tight_layout(pad=2)

    epochs = range(1, len(tacc) + 1)
    ax[0].plot(epochs, tacc, 'r', label='Training accuracy')
    ax[0].plot(epochs, vacc, 'b', label='Validation accuracy')
    ax[0].set_title('Accuracy')
    ax[0].legend()

    ax[1].plot(epochs, tloss, 'r', label='Training loss')
    ax[1].plot(epochs, vloss, 'b', label='Validation loss')
    ax[1].set_title('Loss')
    ax[1].legend()

    plt.show()

else: #if not training
  #load from checkpoint variable
  model.load_weights(cptoload)
  tloss, tacc = model.evaluate(timg, tlabels, verbose=2)
  vloss, vacc = model.evaluate(vimg, vlabels, verbose=2)

  print("MODEL LOAD SUCCESSFUL\n"
  f'checkpoint: {cptoload}\n'
  f'------------------------\n'
  f'train loss: {tloss:>9.3f}\n'
  f'train acc : {tacc:>9.3f}\n'
  f'------------------------\n'
  f'val loss  : {vloss:>9.3f}\n'
  f'val acc   : {vacc:>9.3f}\n')
  f'------------------------\n'

#  calc and print test result
#   prefer not to see this till later
if False:
  eval = model.evaluate(testimg, testlabels)
  print("EVAL\n"
  f'test loss: {eval[0]}\n'
  f'test acc:  {eval[1]}\n')


if LAYEROUTPLOT:
  """
  get layer outputs
  https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
  https://stackoverflow.com/questions/63287641/get-each-layer-output-in-keras-model-for-a-single-image
  
  
  https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
  ^ this one
  """

  #extract layer outputs
  extractor = keras.Model(inputs=model.inputs,
                          outputs=[layer.output for layer in model.layers])
  #features = extractor(timg)

  #plot 1 random images as RGB, including label as true/false 
  fig, ax = plt.subplot_mosaic("AABCD;AAEFG;HIJKL;MNOPQ", figsize=(16,12))
  fig.tight_layout(pad=1)

  rand = np.random.randint(batchlen)   

  img=timg[rand,:,:,:]
  # expand dimensions so that it represents a single 'sample'
  eimg = np.expand_dims(img, axis=0)

  print(eimg.shape)

  feature_maps = extractor.predict(eimg)

  ax["A"].imshow(img)
  ax["A"].set_title(bool(tlabels.numpy()[rand]))
  ax["A"].set_axis_off()

  j=0 #layertoview
  for layer in model.layers:
    # check for convolutional layer
    if ('conv' not in layer.name) and ('pool' not in layer.name):
      j+=1
      continue

    i=0
    for key in ax:
      if key == "A":
        print(layer.name)
        print(j)
      else:
        #print(feature_maps[j][0][:,:,i])
        ax[key].imshow(feature_maps[j][0][:,:,i])
      #  ax[key].set_title(bool(tlabels.numpy()[rand]))
        ax[key].set_axis_off()
        i+=1
    plt.savefig(os.path.join(odir, f'{layer.name}.png'), dpi=300)
    j+=1

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