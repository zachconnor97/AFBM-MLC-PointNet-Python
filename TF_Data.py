import tensorflow as tf
import pandas as pd
import numpy as np

"""
data_file_path = 'test.csv'
username = 'connorz'
cloud_path_header = str('C:/Users/' + username + '/Box/Automated Functional Basis Modeling/ShapeNetCore.v2/AllClouds10k/')
#print(cloud_path_header)
"""

#datafile = tf.keras.utils.get_file("AFBMData_NoChairs.csv")
df = pd.read_csv("AFBMData_NoChairs.csv")
df.head()

# Try this but with our csv file
#x = dict(df)
afbm_slice = tf.data.Dataset.from_tensor_slices(dict(df))

"""
for feature_batch in data_slice.take(1):
  for key, value in feature_batch.items():
    print("  {!r:20s}: {}".format(key, value))
"""