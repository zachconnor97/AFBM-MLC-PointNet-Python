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
import pymeshlab

tf.random.set_seed(1234)
NUM_POINTS = 3000
BATCH_SIZE = 32
NUM_CLASSES = 25
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
        cloud = cloud.uniform_down_sample(every_k_points=2)
    
    except:
        cloud = np.random.rand((NUM_POINTS,3))
        path = 'ERROR IN PCREAD: Transformation from Tensor to String Failed'
        print(path)
    finally:
        #cloud = cloud #np.asarray(cloud) #.points()
        cloud = cloud.points
        cloud = np.asarray([cloud])
    return cloud

# ISparse Matrix Encoding Function
def Sparse_Matrix_Encoding(df):
 
  # Get list/array/whatever of unique labels
  uniquelabels = df.stack().unique()
  uniquelabels.sort()
  uniquelabels = np.delete(uniquelabels,len(uniquelabels)-1,0)
  
  # Encode all of the labels to the point cloud index as a length(dataframe) by length(uniquelabels) sparse matrix (1 or 0 only)
  encodedLabel = np.zeros((len(df), len(uniquelabels)), dtype=int)
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

def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label

def generate_dataset(filename):

    # Import the csv and convert to strings
    df = pd.read_csv("AFBMData_NoChairs.csv")
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

    sparse_matrix = Sparse_Matrix_Encoding(df) 

    # Slice file paths and labels to tf.data.Dataset
    file_paths = np.asmatrix(file_paths)
    nfile_paths = file_paths.reshape((np.size(file_paths),1)) 
    nfile_paths = np.asarray(nfile_paths)
    tfile_paths = tf.constant(nfile_paths.tolist())
    tsparse = tf.constant(sparse_matrix.tolist())
    fileset = tf.data.Dataset.from_tensor_slices((tfile_paths))
    labelset = tf.data.Dataset.from_tensor_slices((tsparse))
    
    train_points = fileset.skip(int(0.3*len(fileset)))
    train_label = labelset.skip(int(.3*len(labelset)))
    
    val_points = fileset.take(int(0.3*len(fileset)))
    val_label = labelset.take(int(.3*len(labelset)))
    
    val_points = val_points.map(lambda x: tf.py_function(pc_read, [x], tf.float64))
    train_points = train_points.map(lambda x: tf.py_function(pc_read, [x], tf.float64))

    
    val_ds = tf.data.Dataset.zip((val_points, val_label))
    train_ds = tf.data.Dataset.zip((train_points, train_label))

    #afbm_dataset = tf.data.Dataset.zip((fileset_new, labelset))

    #train_ds, val_ds = tf.keras.utils.split_dataset(afbm_dataset, left_size=0.7)
    #val_ds = afbm_dataset.take(int(.3*len(afbm_dataset)))
    #train_ds = afbm_dataset.skip(int(0.3*len(afbm_dataset)))

    #val_ds = val_ds.map(lambda x, y: tf.py_function(pc_read, [x, y], tf.float64))
    #train_ds = train_ds.map(lambda x, y: tf.py_function(pc_read, [x, y], tf.float64))

    #val_ds = val_ds.batch(BATCH_SIZE)
    #train_ds = train_ds.batch(BATCH_SIZE)
    #.map(lambda x, y: tf.py_function(augment, [x, y], tf.float64))

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
    return train_ds, val_ds



database = "AFBMData_NoChairs.csv"
#train_ds, val_ds = generate_dataset(filename=database)
cloud = pc_read('testcloud.ply')
print(cloud)
#test = train_ds.take(1)
#points, labels = list(test)[0]
#print(points)