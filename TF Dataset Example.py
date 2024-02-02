import tensorflow as tf
import numpy as np

def read_function(filepath):
    filepath = filepath.numpy()
    filepath = filepath[0]
    data = np.array((filepath, np.random.rand(), 5))
    return data

file_paths = np.random.rand(3000,1)
#print(file_paths)
label_array = np.random.rand(3000,5)

file_paths = np.asarray(file_paths)
file_paths = tf.constant(file_paths.tolist())

fileset = tf.data.Dataset.from_tensor_slices((file_paths))

fileset = fileset.map(lambda x: tf.py_function(read_function, [x], tf.float32))

label_array = tf.constant(label_array.tolist())
labelset = tf.data.Dataset.from_tensor_slices((label_array))

dataset = tf.data.Dataset.zip((fileset, labelset))

data = dataset.take(1)
points, labels = list(data)[0]
print(points.numpy())