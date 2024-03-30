import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import date 
import os
import csv
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import pointnet, generator, OrthogonalRegularizer, orthogonal_regularizer_from_config
from utils import PerLabelMetric, GarbageMan
from dataset_example import generate_dataset
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.src import backend_config
epsilon = backend_config.epsilon
EPS = 1e-7
NUM_POINTS = 5000
NUM_CLASSES = 25
TRAINING = True
LEARN_RATE = 0.25
BATCH_SIZE = 10
NUM_EPOCHS = 18
username = 'Zachariah'
database = "AFBMData_NoChairs_Augmented.csv"
save_path = str('/mnt/c/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/MLCPN_Validation' + str(date.today()) + '_' + str(BATCH_SIZE) + '_' + str(NUM_POINTS) + '_' + str(NUM_EPOCHS) + '_' + 'Learning Rate_' + str(LEARN_RATE) + '_' + 'Epsilon: ' + str(EPS))

label_names = [
    'ConvertCEtoKE,AE,TE', 'ConvertEEtoAE', 'ConvertEEtoLE',
    'ConvertKEtoAE', 'ConvertKEtoEE', 'ConvertLEtoCE',
    'ConvertLEtoEE', 'ExportAE', 'ExportAEtoTE',
    'ExportCE', 'ExportEE', 'ExportGas', 'ExportLE',
    'ExportLiquid', 'ExportSolid', 'ImportEE',
    'ImportGas', 'ImportHE', 'ImportKE',
    'ImportLE', 'ImportLiquid', 'ImportSolid',
    'StoreGas', 'StoreLiquid', 'StoreSolid'
]

g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)
pn_model = pointnet(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=True)

def gradcam_heatcloud(cloud, model, lcln, label_idx=None):
    """
    Inputs:
    cloud: Point Cloud Input. Must be tensor shaped (1, Num Points, Num Dims)
    model: Point Net Model. Takes in point cloud, returns labels
    lcln: Name of the last convolutional layer in PointNet
    label_idx: Index of label to generate heatcloud for
    Need to modify function to accept multiple labels and create multiple intensity vectors
    """
    # from keras.io but need to modify for pcs instead
    gradm = tf.keras.models.Model(
        model.inputs, [model.get_layer(lcln).output, model.output]
    )
    with tf.GradientTape() as tape:
        lclo, preds = gradm(cloud)
        #print(preds)
        if label_idx is None:
            label_idx = tf.argmax(preds[0])
        label_channel = preds[:, label_idx]
    #print(label_idx)
    grads = tape.gradient(label_channel, lclo)
    #print(grads)
    #pooled_grads = tf.reduce_mean(grads, axis=0) # Dimensionality of this var is causing issue in line 190...
    pooled_grads = tf.reduce_mean(grads, axis=(0,1)) 
    #print(pooled_grads[..., tf.newaxis])
    lclo = lclo[0]

    #Checking the shape of the matrices before multipication
    lclo_shape = lclo.shape
    pooled_grads = pooled_grads[..., tf.newaxis]
    #print("Shape of lclo:", lclo_shape)
    #print("Shape of pooled_grads:", pooled_grads.shape)

    # Perform matrix multiplication
    heatcloud = lclo @ pooled_grads
    #heatcloud = lclo @ pooled_grads[..., tf.newaxis] #error here.
    #print(heatcloud.shape)
    heatcloud = tf.squeeze(heatcloud)
    heatcloud = tf.maximum(heatcloud, 0) / tf.math.reduce_max(heatcloud)
    return heatcloud.numpy()

def save_and_display_gradcam(point_cloud, heatcloud, i=None, label_names=None):
    
    pc = point_cloud
    v = np.zeros((len(heatcloud),1))
    #rg[:,0] = np.subtract(rg[:,0], (1 + np.log(heatcloud)))
    #rg[:,1] = np.subtract(rg[:,1], (1 + np.log(heatcloud)))
    #b = np.ones((len(heatcloud),1))

    #g = 1 + np.log(heatcloud)
    g = heatcloud
    g = np.reshape(g, (len(heatcloud),1))
    rgb = np.hstack((v,g,v))
    # Convert back to open3d pc
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc)
    cloud.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([cloud])

    #o3d.io.write_point_cloud("Point_Cloud_Intensity" + label_names[i] + ".ply", cloud)

# Test GradCAM stuff
pn_model.load_weights('MLCPNBestWeights.h5')
pn_model.compile(run_eagerly=True)

"""
#pn_model.summary()
#testcloud = o3d.io.read_point_cloud('C:/Users/gabri/OneDrive - Oregon State University/AllClouds10k/AllClouds10k/lamp_3636649_be13324c84d2a9d72b151d8b52c53b901_10000_2pc.ply') # use open3d to import point cloud from file
#testcloud = o3d.io.read_point_cloud('C:/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AllClouds10k/AllClouds10k/bottle_2876657_2618100a5821a4d847df6165146d5bbd1_10000_2pc.ply') # use open3d to import point cloud from file
#testcloud = o3d.io.read_point_cloud('/mnt/c/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AllClouds10k/AllClouds10k/lamp_3636649_be13324c84d2a9d72b151d8b52c53b901_10000_2pc.ply') # use open3d to import point cloud from file
#pc_path = 'C:/Users/gabri/OneDrive - Oregon State University/AllClouds10k/AllClouds10k/sofa_couch_lounge_4256520_3e3ad2629c9ab938c2eaaa1f79e71ec1_10000_2pc.ply'
pc_path = '/mtn/c/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AllClouds10k/AllClouds10k/lamp_3636649_be13324c84d2a9d72b151d8b52c53b901_10000_2pc.ply' #'sofa_couch_lounge_4256520_3e3ad2629c9ab938c2eaaa1f79e71ec1_10000_2pc.ply'
#bottle_2876657_2618100a5821a4d847df6165146d5bbd1_10000_2pc.ply'
#lamp_3636649_be13324c84d2a9d72b151d8b52c53b901_10000_2pc.ply'
pc = o3d.io.read_point_cloud(pc_path)
pc = pc.uniform_down_sample(every_k_points=2)
pc= pc.points
pc = np.asarray([pc])[0]
testcloud = np.reshape(pc, (1,5000,3))
testcloud = tf.constant(testcloud, dtype='float64')
"""
database = "AFBMData_NoChairs_Augmented.csv"
train_ds, val_ds, label_weights, val_paths = generate_dataset(filename=database)

"""
lln = 'activation_14' #'dot'
labels = pn_model.predict(testcloud)
print("Predicted Labels: ", labels)
pn_model.layers[-1].activation = None

"""
example_clouds = val_ds.take(BATCH_SIZE)
example_clouds = example_clouds.batch(BATCH_SIZE)
example_paths = val_paths.take(BATCH_SIZE)
points, y_true = list(example_clouds)[0]
y_pred = pn_model.predict(example_clouds, batch_size=BATCH_SIZE)

#print(f"Y Predicted: {y_pred}")
print(f"Y Predicted Size: {np.shape(y_pred)}")


lln = 'activation_14' #'dot'
pn_model.layers[-1].activation = None


"""
# Get idx from predicted labels with prediction > 0.5 
y_pred = np.array(y_pred)
pred_label_index = np.where(y_pred >= 0.5)[1]
#print(label_index)

for i in pred_label_index:
    heatcloud = gradcam_heatcloud(testcloud, pn_model, lln, label_idx=i)
    #print(heatcloud)
    save_and_display_gradcam(pc, heatcloud, i, label_names)

"""