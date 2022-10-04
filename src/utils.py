import numpy as np
import matplotlib.pyplot as plt
import yaml


def readcfg(filepath):
    with open(filepath, "r") as f:
        return yaml.safe_load(f)
