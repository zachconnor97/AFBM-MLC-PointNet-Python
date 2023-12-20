import tensorflow as tf
import pandas as pd
import numpy as np

"""
data_file_path = 'test.csv'
username = 'connorz'
cloud_path_header = str('C:/Users/' + username + '/Box/Automated Functional Basis Modeling/ShapeNetCore.v2/AllClouds10k/')
#print(cloud_path_header)
"""

# Import the csv and convert to strings
df = pd.read_csv("AFBMData_NoChairs.csv")
df = df.astype('str')

# Seperates cloud paths to pandas series 
file_paths = df.pop('cloudpath')

# Removes non-necessary dataframe columns to leave just FBM labels
df.pop('SysetID')
df.pop('Name')
df.pop('.obj paths')
df.pop('fileid')
df.pop('status')

# Insert Sparse Matrix Encoding Function
def Sparse_Matrix_Encoding(df):
  # Get list/array/whatever of unique labels
  # uniquelabels = pd.unique(df) # doesn't quite work. pd.concat() might work


  # Encode all of the labels to the point cloud index as a length(dataframe) by length(uniquelabels) sparse matrix (1 or 0 only)
  # Probably be a nested for loop
  # Matlab method 
  """
      encodedLabel = zeros(numClouds,numLabels);
      i=1; j=1; q=1;
      for i = 1:numClouds
          for j = 1:size(labels,2)
              req_index = find(strcmp(string(all_labels), labels{i,j}));
              if req_index
                  if labels{i,j} == "na"
                      encodedLabel(i, req_index) = 0;
                  else
                      encodedLabel(i, req_index) = 1;
                  end
              end
          end
      end"""

  # Placeholder! Remove later
  sparse_matrix = df
  return sparse_matrix

BATCH_SIZE = 64
# Slice file paths and labels to tf.data.Dataset
file_slices = tf.data.Dataset.from_tensor_slices(file_paths).batch(BATCH_SIZE)
label_slices = tf.data.Dataset.from_tensor_slices(dict(df)).batch(BATCH_SIZE)

# How do we get the point clouds into tf.data.Dataset without overflowing memory? 
# Check tf / Keras docs

"""
for feature_batch in data_slice.take(1):
  for key, value in feature_batch.items():
    print("  {!r:20s}: {}".format(key, value))
"""

"""inputs = {}

for name, column in titanic_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)"""