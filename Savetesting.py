import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
import open3d
import pandas as pd

#Creating a test Tensor dataset
dataset1 = tf.data.Dataset.range(0, 3)
print(list(dataset1.as_numpy_iterator()))
dataset1.save('D:\ZachResearch\AFBM-MLC-PointNet-Python\TestDatasets')
dataset2 = tf.data.Dataset.load('D:\ZachResearch\AFBM-MLC-PointNet-Python\TestDatasets')
print(list(dataset2.as_numpy_iterator()))