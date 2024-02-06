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

tf.random.set_seed(1234)
NUM_POINTS = 2000
SAMPLE_RATIO = int(10000 / NUM_POINTS)
print(SAMPLE_RATIO)
BATCH_SIZE = 32
NUM_CLASSES = 25
username = 'Zachariah'

def pc_read(path):
    
    cloud_path_header = str('C:/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AllClouds10k/AllClouds10k/')
    # Use second one for WSL
    #cloud_path_header = str('/mnt/c/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AllClouds10k/AllClouds10k/')
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
train_ds, val_ds = generate_dataset(filename=database)

#save datasets
save_path = str('C:/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AllClouds10k/AllClouds10k/' + date.today() + BATCH_SIZE + NUM_POINTS)
train_path = str(save_path + "train_ds")
train_ds.save(train_path)
val_path = str(save_path + "val_ds")
val_ds.save(val_path)

#afbm_dataset.batch(BATCH_SIZE)

#Seperate into training and validation here:
#train_ds, val_ds = tf.keras.utils.split_dataset(afbm_dataset, left_size=0.8)

train_data = train_ds.take(1)
points, labels = list(train_data)[0]
print(points)

# Sample points down to NUM_POINTS
# Use trimesh probably? 
# Write new function

# Batch data and augment train data. Also Shuffle
#train_data.batch(BATCH_SIZE).map(augment)
#val_data.batch(BATCH_SIZE)

#Also shuffle points
# dataset.shuffle(len(train_points))


"""
train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)
"""

### Build a model

#Each convolution and fully-connected layer (with exception for end layers) consits of
#Convolution / Dense -> Batch Normalization -> ReLU Activation.


def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

"""PointNet consists of two core components. The primary MLP network, and the transformer
net (T-net). The T-net aims to learn an affine transformation matrix by its own mini
network. The T-net is used twice. The first time to transform the input features (n, 3)
into a canonical representation. The second is an affine transformation for alignment in
feature space (n, 3). As per the original paper we constrain the transformation to be
close to an orthogonal matrix (i.e. ||X*X^T - I|| = 0).

"""

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))


def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 64)
    x = conv_bn(x, 128)
    x = conv_bn(x, 1024)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 512)
    x = dense_bn(x, 256)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

"""The main network can be then implemented in the same manner where the t-net mini models
can be dropped in a layers in the graph. Here we replicate the network architecture
published in the original paper but with half the number of weights at each layer as we
are using the smaller 10 class ModelNet dataset.

"""

inputs = keras.Input(shape=(NUM_POINTS, 3))

x = tnet(inputs, 3)
x = conv_bn(x, 64)
x = conv_bn(x, 64)
x = tnet(x, 64)
x = conv_bn(x, 64)
x = conv_bn(x, 128)
x = conv_bn(x, 1024)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 512)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()

"""### Train model

Once the model is defined it can be trained like any other standard classification model
using `.compile()` and `.fit()`.

"""

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
    run_eagerly=True,
)

model.fit(x=train_ds, epochs=5, validation_data=val_ds)

"""## Visualize predictions

We can use matplotlib to visualize our trained model performance.

"""

#data = test_dataset.take(1)

#points, labels = list(data)[0]
#points = points[:8, ...]
#labels = labels[:8, ...]

# run test data through model
#preds = model.predict(points)
#preds = tf.math.argmax(preds, -1)

#points = points.numpy()

# plot points with predicted class and label
#fig = plt.figure(figsize=(15, 10))
#for i in range(8):
#    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
#    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
#    ax.set_title(
#        "pred: {:}, label: {:}".format(
#            CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
#        )
#    )
#    ax.set_axis_off()
#plt.show()