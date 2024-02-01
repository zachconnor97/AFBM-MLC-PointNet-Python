import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import open3d
import pandas as pd

tf.random.set_seed(1234)
NUM_POINTS = 2000
username = 'Zachariah'
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
    #input is a tensor object, needs to become a standard string
    #path = path.numpy().astype('str') #doesn't work
    #path = path.numpy
    
    print(type(path))
    print(path)
    try:
        path.numpy()
    except:
        path = 'Ruh Roh Raggy.txt'
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
fileset_new.map(lambda x: tf.py_function(pc_read, [x], tf.float32)) # map(pc_read)

labelset = tf.data.Dataset.from_tensor_slices((tsparse))
afbm_dataset = tf.data.Dataset.zip((fileset_new, labelset))
afbm_dataset.batch(BATCH_SIZE)

data = afbm_dataset.take(1)
points, labels = list(data)[0]


#DATA_DIR = tf.keras.utils.get_file(
#    "modelnet.zip",
#    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
#    extract=True,
#)
#DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")

"""We can use the `trimesh` package to read and visualize the `.off` mesh files.

"""

#mesh = trimesh.load(os.path.join(DATA_DIR, "chair/train/chair_0001.off"))
#mesh.show() # does not work with WSL 

"""To convert a mesh file to a point cloud we first need to sample points on the mesh
surface. `.sample()` performs a unifrom random sampling. Here we sample at 2048 locations
and visualize in `matplotlib`.

"""

#points = mesh.sample(5000)

#fig = plt.figure(figsize=(5, 5))
#ax = fig.add_subplot(111, projection="3d")
#ax.scatter(points[:, 0], points[:, 1], points[:, 2])
#ax.set_axis_off()
#plt.show()

"""To generate a `tf.data.Dataset()` we need to first parse through the ModelNet data
folders. Each mesh is loaded and sampled into a point cloud before being added to a
standard python list and converted to a `numpy` array. We also store the current
enumerate index value as the object label and use a dictionary to recall this later.

"""
"""
def parse_dataset(num_points=5000):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )
"""
"""Set the number of points to sample and batch size and parse the dataset. This can take
~5minutes to complete.

"""
"""
NUM_POINTS = 5000
NUM_CLASSES = 10
BATCH_SIZE = 32

train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
    NUM_POINTS
)
"""
"""Our data can now be read into a `tf.data.Dataset()` object. We set the shuffle buffer
size to the entire size of the dataset as prior to this the data is ordered by class.
Data augmentation is important when working with point cloud data. We create a
augmentation function to jitter and shuffle the train dataset.

"""
"""
def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

"""### Build a model

#Each convolution and fully-connected layer (with exception for end layers) consits of
#Convolution / Dense -> Batch Normalization -> ReLU Activation.

"""
"""
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

""" We can then define a general function to build T-net layers.

"""

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
NUM_CLASSES = 25
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
)

model.fit(data, epochs=5)

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