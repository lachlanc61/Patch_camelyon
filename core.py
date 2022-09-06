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
PREPLOT=False     #plot some images before running model
POSTPLOT=True     #plot accuracy and loss over time

batchlen = 256    #size of batch for fitting
buffer=5000       #buffer for shuffling

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
model = Sequential(
    [
        keras.layers.Flatten(input_shape=(96, 96, 3)),
 #      tf.keras.Input(shape=(96,96,3)),    #specify input size
        ### START CODE HERE ### 
        keras.layers.Dense(25, activation='sigmoid'),
        keras.layers.Dense(15, activation='sigmoid'), 
        keras.layers.Dense(1, activation='sigmoid') 
        ### END CODE HERE ### 
    ], name = "my_model" 
)   

#view model summary
model.summary()

#print parameters
L1_num_params = 400 * 25 + 25  # W1 parameters  + b1 parameters
L2_num_params = 25 * 15 + 15   # W2 parameters  + b2 parameters
L3_num_params = 15 * 1 + 1     # W3 parameters  + b3 parameters
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params, ",  L3 params = ", L3_num_params )

[layer0, layer1, layer2, layer3] = model.layers

#### Examine Weights & shapes
W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

#compile the model
#   acc = track accuracy vs train/val sets
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['acc'],
)

#run the fit, saving params to fitlog 
#   epochs = N. cycles
#   validation data is tested each epoch
fitlog = model.fit(
    timg,tlabels,
    epochs=50,
    validation_data = (vimg, vlabels), verbose=1
)

#extract metrics for plotting
tacc = fitlog.history['acc']
tloss = fitlog.history['loss']
vacc = fitlog.history['val_acc']
vloss = fitlog.history['val_loss']

#evaluate model against reserved test data
eval = model.evaluate(testimg, testlabels)

print(
  "EVAL\n"
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

print("CLEAN EXIT")
exit()
