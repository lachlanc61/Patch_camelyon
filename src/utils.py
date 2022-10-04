import yaml
import os
import sys


import tensorflow as tf


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