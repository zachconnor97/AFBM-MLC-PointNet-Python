import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import date 
import csv
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorsys
import matplotlib
from model import pointnet, OrthogonalRegularizer, orthogonal_regularizer_from_config
from utils import PerLabelMetric, GarbageMan
from dataset_example import generate_dataset
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend_config

#a = tf.random.normal(shape=[10,5], dtype=tf.float32)
#b = tf.random.normal(shape=[5,10], dtype=tf.float32)
a = tf.constant([[1,1,1,1]])
b = tf.constant([[1],[1],[1],[1]])
c = tf.matmul(a,b)
print(c)