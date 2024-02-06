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
from datetime import date

print("Today is: ",date.today())

#Creating a test Tensor dataset
dataset1 = tf.data.Dataset.range(0, 9)
print(list(dataset1.as_numpy_iterator()))
Path = str('D:\ZachResearch\AFBM-MLC-PointNet-Python\TestDatasets' + "Test_file")
dataset1.save(Path)
dataset2 = tf.data.Dataset.load(Path)
print(list(dataset2.as_numpy_iterator()))