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

physical_devices = tf.config.list_physical_devices('GPU')
#print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.random.set_seed(1234)
NUM_POINTS = 2000
SAMPLE_RATIO = int(10000 / NUM_POINTS)
print("Sample Ratio:")
print(1/SAMPLE_RATIO)
BATCH_SIZE = 32
NUM_CLASSES = 25
NUM_EPOCHS = 20
LEARN_RATE = 0.00025
username = 'Zachariah'

class GarbageMan(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

# Custom Label Metric
class PerLabelMetric(Metric):
    def __init__(self,name='per_label_metric', num_labels=NUM_CLASSES, **kwargs):
        super(PerLabelMetric, self).__init__(name=name,**kwargs)
        self.num_labels = num_labels
        self.true_positives = self.add_weight(name='true_positives', shape=(self.num_labels), initializer='zeros')
        self.true_negatives = self.add_weight(name='true_negatives', shape=(self.num_labels), initializer='zeros')
        self.false_positives = self.add_weight(name='false_positives', shape=(self.num_labels), initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', shape=(self.num_labels), initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Custom logic to compute the metric for each label
        for i in range(self.num_labels):
            y_true_label = y_true[:, i]
            y_pred_label = y_pred[:, i]
            
            true_positives = B.sum(B.cast(y_true_label * B.round(y_pred_label), 'float32'))
            false_positives = B.sum(B.cast((1 - y_true_label) * B.round(y_pred_label), 'float32'))
            true_negatives = B.sum(B.cast((1 - y_true_label) * (1 - B.round(y_pred_label)), 'float32'))
            false_negatives = B.sum(B.cast(y_true_label * (1 - B.round(y_pred_label)), 'float32'))
            print(self.true_positives[i].numpy())
            self.true_positives[i].__add__(true_positives)
            self.false_positives[i].__add__(false_positives)
            self.true_negatives[i].__add__(true_negatives)
            self.false_negatives[i].__add__(false_negatives)

    def result(self):
        #precision = self.true_positives / (self.true_positives + self.false_positives + B.epsilon())
        #recall = self.true_positives / (self.true_positives + self.false_negatives + B.epsilon())
        tp = self.true_positives
        tn = self.true_negatives
        fp = self.false_positives
        fn = self.false_negatives
        return tp.numpy(), tn.numpy(), fp.numpy(), fn.numpy()

    def reset_states(self):
        # Reset the state of the metric
        B.batch_set_value([(v, 0) for v in self.variables])

labels = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']

class PerLabelMetricCallBack(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, batch, epoch, logs=None):
        data = self.test_data
        x_data, y_data = data
        correct = 0
        incorrect = 0
        x_result = self.model.predict(x_data, verbose=0)
        x_numpy = []
        for i in labels:
            self.class_history.append([])
        class_correct = [0] * len(labels)
        class_incorrect = [0] * len(labels)
        for i in range(len(x_data)):
            x = x_data[i]
            y = y_data[i]
            res = x_result[i]
            actual_label = np.argmax(y)
            pred_label = np.argmax(y)
            if(pred_label == actual_label):
                x_numpy.append(["cor:", str(y), str(res), str(pred_label)])
                class_correct[actual_label] += 1
                correct += 1
            else:
                x_numpy.append(["inc:", str(y), str(res), str(pred_label)])
                class_incorrect[actual_label] += 1
                incorrect += 1
        print("\tCorrect: %d" %(correct))
        print("\tIncorrect: %d" %(incorrect))
        for i in range(len(labels)):
            tot = float(class_correct[i] + class_incorrect[i])
            class_acc = -1
            if (tot > 0):
                class_acc = float(class_correct[i]) / tot
            print("\t%s: %.3f" %(class_correct[i], class_acc))
        acc = float(correct) / float(correct + incorrect)
        print("\tCurrent Network Accuracy: %.3f" %(acc))

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
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
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

database = "AFBMData_NoChairs_Augmented.csv"
train_ds, val_ds, label_weights = generate_dataset(filename=database)
#print(type(label_weights))
#print("\tLabel Weights: %d",label_weights)

with open("Label_Weights.csv", mode='w') as f:
    writer = csv.writer(f)
    for key, value in label_weights.items():
        writer.writerow([key, value])

#save_path = str('C:/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/' + str(date.today()) + '_' + str(BATCH_SIZE) + '_' + str(NUM_POINTS))
save_path = str('/mnt/c/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/' + str(date.today()) + '_' + str(BATCH_SIZE) + '_' + str(NUM_POINTS) + '_' + str(NUM_EPOCHS) + '_' + 'Learning Rate_' + str(LEARN_RATE))

train_path = str(save_path + "train_ds")
val_path = str(save_path + "val_ds")
#train_ds.save(train_path)
#val_ds.save(val_path)


#load_path = "C:/Users/" + username + "/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/"
load_path = "/mnt/c/Users/" + username + "/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/"
#train_ds = tf.data.Dataset.load(load_path + '2024-02-07_32_2000train_ds')
#val_ds = tf.data.Dataset.load(load_path + '2024-02-07_32_2000val_ds')

"""
train_data = train_ds.take(2)
for batch in range(len(train_data)):
    points, labels = list(train_data)[batch]
    print(labels.numpy())
    print(labels)
    #print(points.numpy().max())
    #print(points.numpy())
"""

### PointNet Model
#Each convolution and fully-connected layer (with exception for end layers) consits of
#Convolution / Dense -> Batch Normalization -> ReLU Activation.

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation('LeakyReLU')(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation('LeakyReLU')(x)

#PointNet consists of two core components. The primary MLP network, and the transformer
#net (T-net). The T-net aims to learn an affine transformation matrix by its own mini
#network. The T-net is used twice. The first time to transform the input features (n, 3)
#into a canonical representation. The second is an affine transformation for alignment in
#feature space (n, 3). As per the original paper we constrain the transformation to be
#close to an orthogonal matrix (i.e. ||X*X^T - I|| = 0).

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features=None, l2reg=0.001, **kwargs):
        super(OrthogonalRegularizer, self).__init__(**kwargs)
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
        return{'num_features': self.num_features,'l2reg': self.l2reg}

    @classmethod
    def from_config(cls, config):
        return cls(num_features=config.pop('num_features', None), **config)
    
## Custom function to instantiate the regularizer during loading
def orthogonal_regularizer_from_config(config):
    return OrthogonalRegularizer(**config)

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

#The main network can be then implemented in the same manner where the t-net mini models
#can be dropped in a layers in the graph. Here we replicate the network architecture
#published in the original paper but with half the number of weights at each layer as we
#are using the smaller 10 class ModelNet dataset.

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
#x = layers.Flatten()(x)
#x = layers.Dropout(0.3)(x)
x = dense_bn(x, 256)
#x = layers.Flatten()(x)
#x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
#model.summary()

### Train model

#Once the model is defined it can be trained like any other standard classification model
#using `.compile()` and `.fit()`.
#acc_per_label = PerLabelMetricCallBack(val_ds)
"""
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=LEARN_RATE),
    metrics=[
            #PerLabelMetric(num_labels=NUM_CLASSES),
            tf.keras.metrics.BinaryAccuracy(threshold=0.5),
            tf.keras.metrics.Precision(thresholds=[0.5,1]),
            tf.keras.metrics.Recall(thresholds=[0.5,1]),
            tf.keras.metrics.F1Score(threshold=0.5)
            ],      
    run_eagerly=True,
)
"""
## Save Model
#model.save(save_path + '_AFBM Model')
## Load Model here
keras.utils.get_custom_objects()['OrthogonalRegularizer'] = OrthogonalRegularizer
model = tf.keras.models.load_model("/mnt/c/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/2024-02-16_32_2000_20_Learning Rate_0.00025_AFBM Model", custom_objects={'OrthogonalRegularizer': orthogonal_regularizer_from_config})  #save_path + '_AFBM Model', custom_objects={'OrthogonalRegularizer': orthogonal_regularizer_from_config})
#model = tf.keras.models.load_model(save_path + '_AFBM Model')
#model.summary()

#train_hist = model.fit(x=train_ds, epochs=NUM_EPOCHS, class_weight=label_weights, validation_data=val_ds, callbacks=[GarbageMan()])
#model.fit(x=val_ds, epochs=NUM_EPOCHS, class_weight=label_weights, validation_data=val_ds, callbacks=[GarbageMan()])
#model.evaluate(x=val_ds,callbacks=[acc_per_label])
#model.save(save_path + '_AFBM Model')

"""
## Save history file
histdf = pd.DataFrame(train_hist.history)
histfile = save_path + '_train_history2.csv'
with open(histfile, mode='w') as f:
    histdf.to_csv(f)
"""
    
# Validation / Evaluation per Label
data = []
for i in range(0,NUM_CLASSES):
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=LEARN_RATE),
        metrics=[
            PerLabelMetric(num_labels=NUM_CLASSES),
            tf.keras.metrics.Precision(thresholds=[0.5, 1],class_id=i),
            tf.keras.metrics.Recall(thresholds=[0.5, 1],class_id=i),
            tf.keras.metrics.F1Score(threshold=0.5),      
        ],
        run_eagerly=True,
    )
    data.append(model.evaluate(x=val_ds))
    
histdf = pd.DataFrame(data)
histfile = save_path + '_label_validation_Testing.csv'
with open(histfile, mode='w') as f:
    histdf.to_csv(f)


#model.summary()
## Save Model
#model.save(save_path + '_AFBM Model')
## Load Model here
#keras.utils.get_custom_objects()['OrthogonalRegularizer'] = OrthogonalRegularizer
#model = tf.keras.models.load_model(save_path + '_AFBM Model', custom_objects={'OrthogonalRegularizer': orthogonal_regularizer_from_config})
## Test if the loaded model is the same
#model.summary()









"""
# Visualize predictions

#We can use matplotlib to visualize our trained model performance.
data = train_ds.take(1)
points, labels = list(data)[0]
#points = points[:8, ...]
#labels = labels[:8, ...]

# run test data through model
preds = model.predict(points)
preds = tf.math.argmax(preds, -1)

points = points.numpy()

# plot points with predicted class and label
fig = plt.figure(figsize=(15, 10))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
    ax.set_title(
        "pred: {:}, label: {:}".format(
            CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
        )
    )
    ax.set_axis_off()
plt.show()
"""
