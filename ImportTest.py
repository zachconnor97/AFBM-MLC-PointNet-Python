import tensorflow as tf
import pandas as pd
import numpy as np
import open3d
 
username = 'Zachariah'

def pc_read(path):
    cloud_path_header = str('C:/Users/' + username + '/Box/Automated Functional Basis Modeling/ShapeNetCore.v2/AllClouds10k/')
    try:
        path = path.numpy()
        path = np.array2string(path)
        for character in "[]]b'":
            path = path.replace(character, '')
    except:
        path = 'ERROR IN PCREAD: Transformation from Tensor to String Failed'
        print(path)
    finally:
        print(path)
        print(type(path))
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

def generate_dataset(username, batch_size):

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
    BATCH_SIZE = 64

    # Slice file paths and labels to tf.data.Dataset
    file_paths = np.asmatrix(file_paths)
    nfile_paths = file_paths.reshape((np.size(file_paths),1)) 
    nfile_paths = np.asarray(nfile_paths)
    tfile_paths = tf.constant(nfile_paths.tolist())
    tsparse = tf.constant(sparse_matrix.tolist())
    fileset_new = tf.data.Dataset.from_tensor_slices((tfile_paths))

    fileset_new = fileset_new.map(lambda x: tf.py_function(pc_read, [x], tf.float64))

    labelset = tf.data.Dataset.from_tensor_slices((tsparse))
    afbm_dataset = tf.data.Dataset.zip((fileset_new, labelset))

    #Testing stuff
    
    data = afbm_dataset.take(299)
    points, labels = list(data)[0]
    print(points.numpy())
    print(type(points.numpy()))
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points.numpy())
    open3d.visualization.draw_geometries([pcd])
    
    return afbm_dataset

if __name__=="__main__":
    afbm_dataset = generate_dataset(username=username,batch_size=64)