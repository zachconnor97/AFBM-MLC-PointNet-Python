import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
import open3d
import pandas as pd
from datetime import date 
import gc
from tensorflow.keras.metrics import Metric
from keras import backend as B
import csv


NUM_POINTS = 5000
SAMPLE_RATIO = int(10000 / NUM_POINTS)
BATCH_SIZE = 16
username = 'Zachariah'

def pc_read(path):
    
    #cloud_path_header = str('C:/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AllClouds10k/AllClouds10k/')
    # Use second one for WSL
    cloud_path_header = str('/mnt/c/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AllClouds10k/AllClouds10k/')
    try:
        path = path.numpy()
        path = np.array2string(path)
        for character in "[]]'":
            path = path.replace(character, '')
        path = path[1:]
        path = cloud_path_header + path 
        cloud = open3d.io.read_point_cloud(path)
        cloud = cloud.uniform_down_sample(every_k_points=int(SAMPLE_RATIO))
    except:
        cloud = np.random.rand((NUM_POINTS,3))
        path = 'ERROR IN PCREAD: Transformation from Tensor to String Failed'
        print(path)
    finally:
        cloud = cloud.points
        cloud = np.asarray([cloud])[0]
    
    return cloud

# ISparse Matrix Encoding Function
def Sparse_Matrix_Encoding(df):
 
  # Get list/array/whatever of unique labels
  uniquelabels = df.stack().unique()
  uniquelabels.sort()
  uniquelabels = np.delete(uniquelabels,len(uniquelabels)-1,0)
  
  # Encode all of the labels to the point cloud index as a length(dataframe) by length(uniquelabels) sparse matrix (1 or 0 only)
  encodedLabel = np.zeros((len(df), len(uniquelabels)), dtype=float)
  # Loop through clouds and labels
  for i in range(len(df)):
      for j, label in enumerate(df.columns):
          req_index = np.where(uniquelabels == df.iloc[i, j])[0]
          if req_index.size > 0:
              req_index = req_index[0]
              if df.iloc[i, j] == "nan":
                  encodedLabel[i, req_index] = 0
              else:
                  encodedLabel[i, req_index] = 1
  sparse_matrix = encodedLabel
  return sparse_matrix

def augment(points):
    try:
        # jitter points
        points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
        # shuffle points
        points = tf.random.shuffle(points)
    except:
        points = points
    return points

def generate_dataset(filename):

    # Import the csv and convert to strings
    df = pd.read_csv(filename)
    df = df.astype('str')
    
    # Seperates cloud paths to pandas series 
    file_paths = df.pop('cloudpath')
    #print(file_paths)
    # Removes non-necessary dataframe columns to leave just FBM labels
    df.pop('SysetID')
    df.pop('Name')
    df.pop('.obj paths')
    df.pop('fileid')
    df.pop('status')
    num_files = float(len(df))
    sparse_matrix = Sparse_Matrix_Encoding(df) 
    df = []
    label_counts = sparse_matrix.sum(axis=0)
    label_weights = (num_files / (25 * label_counts))
    label_weights = {k: v for k, v in enumerate(label_weights)}
    #print(type(label_weights))
    #print(label_weights)

    # Slice file paths and labels to tf.data.Dataset
    file_paths = np.asmatrix(file_paths)
    nfile_paths = file_paths.reshape((np.size(file_paths),1)) 
    nfile_paths = np.asarray(nfile_paths)
    tfile_paths = tf.constant(nfile_paths.tolist())
    tsparse = tf.constant(sparse_matrix.tolist())
    fileset = tf.data.Dataset.from_tensor_slices((tfile_paths))
    labelset = tf.data.Dataset.from_tensor_slices((tsparse))
    
    train_points = fileset.skip(int(0.3*len(fileset)))
    train_label = labelset.skip(int(0.3*len(labelset)))
    
    val_points = fileset.take(int(0.3*len(fileset)))
    val_label = labelset.take(int(0.3*len(labelset)))
    
    val_points = val_points.map(lambda x: tf.py_function(pc_read, [x], tf.float64))
    train_points = train_points.map(lambda x: tf.py_function(pc_read, [x], tf.float64))
    train_points = train_points.map(lambda x: tf.py_function(augment, [x], tf.float64))

    #val_ds = tf.data.Dataset.zip((val_points, val_label))
    #train_ds = tf.data.Dataset.zip((train_points, train_label))
    val_ds = tf.data.Dataset.zip((val_points, val_label))
    train_ds = tf.data.Dataset.zip((train_points, train_label))
    val_ds = val_ds.batch(BATCH_SIZE)
    train_ds = train_ds.batch(BATCH_SIZE) # ADDS A lot of time .shuffle(buffer_size=20000,reshuffle_each_iteration=True)

    #Testing stuff
    """
    data = afbm_dataset.take(1)
    points, labels = list(data)[0]
    #print(labels)
    print(points.numpy())
    print(type(points.numpy()))
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points.numpy())
    open3d.visualization.draw_geometries([pcd])
    """
    return train_ds, val_ds, label_weights