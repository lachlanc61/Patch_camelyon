import numpy as np
import matplotlib.pyplot as plt
import sys

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
PREPLOT=False       #plot some images before running model
POSTPLOT=True       #plot accuracy and loss over time
LAYEROUTPLOT=False  #plot layer outputs - not working yet

batchlen = 512    #size of batch for fitting
buffer=5000       #buffer for shuffling
nepochs=100       #no. epochs
stoplim=25      #patience for early stop

lamda=0.0001
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

print(f'python version: {sys.version}')
print(f'tensorflow version: {tf.__version__}')

gpu_available = (len(tf.config.list_physical_devices('GPU')) > 0)
print(f'GPU: {gpu_available}')

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

#Initialise basic TF model
#   flatten RGB images as first layer
#
#   use relu function instead of signmoid for all but output layer
#   basically max(0,val)
#   = passthrough if above 0
#     more responsive across entire range
#     sigmoid only really sensitive around inflection point. high always ->1, low always->0
#   add two more layers

l2reg=tf.keras.regularizers.L2(lamda)
#kernel_regularizer=l2reg

model = Sequential(
    [
        keras.layers.Conv2D(32,4, activation='relu', input_shape=[96, 96, 3]),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
       # keras.layers.Flatten(input_shape=(96, 96, 3)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(56, activation='relu'),
        keras.layers.Dense(12, activation='relu'), 
        keras.layers.Dense(1, activation='sigmoid') 
        ### END CODE HERE ### 
    ], name = "my_model" 
)   


#view model summary
model.summary()

#compile the model
#   acc = track accuracy vs train/val sets
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.0005),
    metrics=['acc'],
)

# add a callback to stop refinement early if train loss stops improving
#   maybe better to do this on val? not sure
stopcond=keras.callbacks.EarlyStopping(monitor='val_loss', patience=stoplim)

#run the fit, saving params to fitlog 
#   epochs = N. cycles
#   validation data is tested each epoch
fitlog = model.fit(
    timg,tlabels,
    epochs=nepochs,
    validation_data = (vimg, vlabels),
    callbacks=[stopcond],
    verbose=1
)

#extract metrics for plotting
tacc = fitlog.history['acc']
tloss = fitlog.history['loss']
vacc = fitlog.history['val_acc']
vloss = fitlog.history['val_loss']

#  calc and print test result
#   prefer not to see this till later
if False:
  eval = model.evaluate(testimg, testlabels)
  print("EVAL\n"
  f'test loss: {eval[0]}\n'
  f'test acc:  {eval[1]}\n')

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


if LAYEROUTPLOT:
  """
  attempting to get layer outputs - not sure what i'm looking at yet
  https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
  https://stackoverflow.com/questions/63287641/get-each-layer-output-in-keras-model-for-a-single-image
  """
  extractor = keras.Model(inputs=model.inputs,
                          outputs=[layer.output for layer in model.layers])
  features = extractor(timg)
  print(features[1])
  print(features[1].shape[0])
  print(features[1].shape[1])

  plt.imshow(features[2])
  plt.show()

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
  so what is this?

  
"""

