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
def pc_read(path,label):
    #input is a tensor object, needs to become a standard string
    #path = path.numpy().astype('str') #doesn't work
    #path = path.numpy
    
    print(type(path))
    print(path)
    try:
        path.numpy()
    except:
        path = 'Ruh Roh Raggy'
    finally:
        print(path)
        print(type(path))
    path = cloud_path_header + path 
    cloud = open3d.io.read_point_cloud(path)
    """
    open3d.visualization.draw_geometries([cloud])
    cloud = cloud.voxel_down_sample(voxel_size=0.05)
    open3d.visualization.draw_geometries([cloud])
    """
    cloud = np.asarray(cloud.points)
    return cloud

#cloud1 = pc_read("10155655850468db78d106ce0a280f871.ply")
#print(type(cloud1))

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
file_paths = np.asmatrix(file_paths)
nfile_paths = file_paths.reshape((np.size(file_paths),1)) 
nfile_paths = np.asarray(nfile_paths)
tfile_paths = tf.constant(nfile_paths.tolist())
tsparse = tf.constant(sparse_matrix.tolist())
fileset_new = tf.data.Dataset.from_tensor_slices((tfile_paths))
fileset_new.map(lambda x: tf.py_function(func=pc_read, inp=[x], Tout=tf.float32))

labelset = tf.data.Dataset.from_tensor_slices((tsparse))
afbm_dataset = tf.data.Dataset.zip((fileset_new, labelset))

data = afbm_dataset.take(1)
points, labels = list(data)[0]
print(points)
print(labels)

#points = points[:8, ...]
#labels = labels[:8, ...]

#for element in afbm_dataset.as_numpy_iterator():
#    print(element[0])


"""
sparse_matrix = np.asarray(sparse_matrix.astype('str'))
zata = np.concatenate((nfile_paths,sparse_matrix),axis=1)

ten_size = zata.shape
print(ten_size[0])

# NUMPY list not good for tensorflow 
# https://stackoverflow.com/questions/58636087/tensorflow-valueerror-failed-to-convert-a-numpy-array-to-a-tensor-unsupporte

#train_data = tf.data.Dataset.from_tensor_slices(zata)
#train_data.map(pc_read)
tdata = tf.TensorArray(dtype=tf.string, size=ten_size[0], dynamic_size=True, clear_after_read=False)
zata = tf.constant(zata, dtype=tf.string)

for i in range(ten_size[0]):
    tdata = tdata.write(i,zata[i])

train_data = tf.data.Dataset.from_tensor_slices(tdata)
train_data.map(pc_read)

#for i in range(10):
#    print(tdata.read(i).numpy())
"""
"""
for element in tdata:
    print(element)
"""
#split to training and validation sets
#val_data = train_data.map()
#train_data = train_data.map()

# Test that the read function is working as intended
# Visualize ONE point cloud to verify this is working
# Be careful with this forloop
#for element in train_data:
#    print(element)

#open3d.visualization.draw_geometries([cloud1])
    
#cloud1 = pc_read(file_paths(0))
