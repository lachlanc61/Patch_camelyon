import yaml
import os
import sys


import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from keras import Sequential 

#-----------------------------------
#FUNCTIONS
#-----------------------------------

def readcfg(filepath):
    with open(filepath, "r") as f:
        return yaml.safe_load(f)

def initialise(config):
    #initialise directories relative to script
    script = os.path.realpath(__file__) #_file = current script
    spath=os.path.dirname(script) 
    spath=os.path.dirname(spath)    #second call to get src/..
    wdir=os.path.join(spath,config['wdirname'])
    odir=os.path.join(spath,config['odirname'])

    checkpoint_path = os.path.join(wdir,"cp-{epoch:04d}.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)

    print("script path:", spath)
    print("working path:", wdir)
    print("output path:", odir)
    print("cp:", checkpoint_path)
    print("cp:", checkpoint_dir)
    print("---------------")

    #if we're not training, hide GPU to avoid it going OOM when viewing layer outputs
    #   sure there must be a way to do this more cleanly
    if not config['DOTRAIN']:
        tf.config.set_visible_devices([], 'GPU')

    print(f'python version: {sys.version}')
    print(f'tensorflow version: {tf.__version__}')

    gpu_available = (len(tf.config.list_physical_devices('GPU')) > 0)
    print(f'GPU: {gpu_available}')

    return spath, wdir, odir, checkpoint_path, checkpoint_dir

def normalise(img, labels):
  """
  simple normaliser to pass to tfds.map
    scales 0-255 to 0-1
    receives tensor tuple, returns same 
  """
  img = tf.cast(img, tf.float32)
  img = img/255.
  return img, labels

def rotateflip(ds):
  """
  augments dataset by rotation/flipping

  https://www.tensorflow.org/guide/gpu
  manually placed on CPU because fails on GPU... not clear why

  would prefer to place in primary Sequential model but can't put on CPU there
  eg. https://www.tensorflow.org/tutorials/images/data_augmentation
  """
  #define the augmentations
  augmentation = Sequential(
    [
        keras.layers.RandomFlip("horizontal_and_vertical", input_shape = (96, 96, 3)),
        keras.layers.RandomRotation(0.2),
    ]
  )
  #map augmentation to each image in dataset
  ds=ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

  #return dataset with prefetch
  return ds

def datainit(config, dsetname):
  print("---------------")
  #load the dataset
  with tf.device('/CPU:0'):

    data, dsinfo = tfds.load(dsetname, with_info = True, as_supervised = True)

    dtrain = data['train']
    dvalidation = data['validation']
    dtest = data['test']

    #normalise all to range(0,1) - speeds up ML calc
    # tfds.map performs (function) on each element of the array
    dtrain = dtrain.map(normalise)
    dvalidation = dvalidation.map(normalise)
    dtest = dtest.map(normalise)

    #shuffle the training data to use a different set each time
    dtrain=dtrain.shuffle(config['buffer'])

    #break into sub-batches for training
    dtrain = dtrain.batch(config['batchsize']) #.prefetch(1)
    dvalidation = dvalidation.batch(config['batchsize']) #.prefetch(1)
    dtest = dtest.batch(config['batchsize']) #.prefetch(1)

    #apply augmentations to train only
    #   these objects seem to work like generators - ie. rotateflip is reapplied to each 
    dtrain = rotateflip(dtrain)

    dtrain=dtrain.prefetch(1)
    
    #repeat step loops generator within epoch
    #from https://stackoverflow.com/questions/55421290/tensorflow-2-0-keras-how-to-write-image-summaries-for-tensorboard/55754700#55754700
    #does not seem to work as intended
    #dtrain = dtrain.repeat(config['nepochs'])

  return dtrain, dvalidation, dtest, dsinfo


def batchcheck(dtrain, dvalidation, dtest, batchsize):
  """
  prints diams of first batch
  """
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
  print("batch size:",batchsize)

  return timg, tlabels, vimg, vlabels, testimg, testlabels


def getlen(dtrain, dsinfo):
  """
  report length of dataset
  """
  print("from data:")
  print(dtrain.cardinality().numpy())
  print("from info:")
  print(dsinfo.splits['train'].num_examples)