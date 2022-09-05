import time
import os

import config
import localconf

def startup():
  """
  initialisation routine
  """

  if config.LOCALDAT == True:   
      wdirname=config.ddirname   
  else:
      wdirname=localconf.extdirname

  #initialise directories relative to script
  scriptdir = os.path.realpath(__file__)    #_file = current script
  spath=os.path.dirname(scriptdir)          #path_to_script
  wdir=os.path.join(spath,wdirname)         #working directory
  odir=os.path.join(spath,config.odirname)  #output directory
  print("running:", scriptdir)
  print("---------------")  


def timed(f):
  """
  measures time to run function f
    returns tuple of (output of function), time

  call as: 
    out, runtime=timed(lambda: gapfill2(data))
  
  https://stackoverflow.com/questions/5478351/python-time-measure-function
  """
  start = time.time()
  ret = f()
  elapsed = time.time() - start
  return ret, elapsed
