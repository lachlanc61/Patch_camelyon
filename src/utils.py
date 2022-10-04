import yaml
import os
import sys


import tensorflow as tf
import tensorflow_datasets as tfds

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


def datainit(config, dsetname):
  print("---------------")
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

  #get a sub-batch for training
  dtrain = dtrain.batch(config['batchlen']) 
  dvalidation = dvalidation.batch(config['batchlen']) 
  dtest = dtest.batch(config['batchlen']) 

  return dtrain, dvalidation, dtest, dsinfo


def tensorinit(dtrain, dvalidation, dtest):
  #extract tensor elements
  #   iter converts to iterable object
  #   next extracts element from each
  #   comes as (image, label) tuple
  timg, tlabels = next(iter(dtrain))
  vimg, vlabels = next(iter(dvalidation))
  testimg, testlabels  = next(iter(dtest))

  return timg, tlabels, vimg, vlabels, testimg, testlabels
