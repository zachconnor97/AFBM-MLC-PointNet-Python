import tensorflow as tf
import pandas as pd
import numpy as np
import open3d
 
username = 'Zachariah Connor'
cloud_path_header = str('C:/Users/' + username + '/Box/Automated Functional Basis Modeling/ShapeNetCore.v2/AllClouds10k/')
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
 
# Point Cloud Read Function
def pc_read(path):
    path = cloud_path_header + path
    cloud = open3d.io.read_point_cloud(path)
    cloud = np.asarray(cloud.points)
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

sparse_matrix = Sparse_Matrix_Encoding(df) 
BATCH_SIZE = 64
# Slice file paths and labels to tf.data.Dataset
#print(type(file_paths))
# np.asarray(file_paths)
file_paths = np.asmatrix(file_paths)
nfile_paths = file_paths.reshape((np.size(file_paths),1)) 
nfile_paths = np.asarray(nfile_paths)
sparse_matrix = np.asarray(sparse_matrix.astype('str'))
zata = np.concatenate((nfile_paths,sparse_matrix),axis=1)
print(zata)

# NUMPY list not good for tensorflow 
# https://stackoverflow.com/questions/58636087/tensorflow-valueerror-failed-to-convert-a-numpy-array-to-a-tensor-unsupporte

#train_data = tf.data.Dataset.from_tensor_slices(nfile_paths).batch(BATCH_SIZE)
train_data = tf.data.Dataset.from_tensor_slices(zata).batch(BATCH_SIZE)
train_data.map(lambda name: tf.py_function(pc_read, [name], tf.int32))


for element in train_data:
    print(element)

#cloud1 = pc_read(file_paths(0))
#open3d.visualization.draw_geometries([cloud1])