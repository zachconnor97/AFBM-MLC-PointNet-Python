import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import date 
import csv
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt
from model import pointnet, OrthogonalRegularizer, orthogonal_regularizer_from_config
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
BATCH_SIZE = 500
NUM_EPOCHS = 18
username = 'Zachariah'
database = "AFBMData_NoChairs_Augmented.csv"
save_path = str('C:/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/MLCPN_Validation_dense_8' + str(date.today()) + '_' + str(BATCH_SIZE) + '_' + str(NUM_POINTS) + '_' + str(NUM_EPOCHS) + '_' + 'Learning Rate_' + str(LEARN_RATE) + '_' + 'Epsilon_' + str(EPS))

label_names = [
    'ConvertCEtoKE,AE,TE', 'ConvertEEtoAE', 'ConvertEEtoLE',
    'ConvertKEtoAE', 'ConvertKEtoEE', 'ConvertLEtoCE',
    'ConvertLEtoEE', 'ExportAE', 'ExportAE,TE',
    'ExportCE', 'ExportEE', 'ExportGas', 'ExportLE',
    'ExportLiquid', 'ExportSolid', 'ImportEE',
    'ImportGas', 'ImportHE', 'ImportKE',
    'ImportLE', 'ImportLiquid', 'ImportSolid',
    'StoreGas', 'StoreLiquid', 'StoreSolid'
]

g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)

def gradcam_heatcloud(cloud, model, lcln, label_idx=None):
    """
    Inputs:
    cloud: Point Cloud Input. Must be tensor shaped (1, Num Points, Num Dims)
    model: Point Net Model. Takes in point cloud, returns labels
    lcln: Name of the last convolutional layer in PointNet
    label_idx: Index of label to generate heatcloud for
    """
    cloud = np.reshape(cloud, (1, NUM_POINTS, 3))
    gradm = tf.keras.models.Model(
        model.inputs, [model.get_layer(lcln).output, model.output]
    )
    with tf.GradientTape() as tape:
        lclo, preds = gradm(cloud)
        if label_idx is None:
            label_idx = tf.argmax(preds[0])
        label_channel = preds[:, label_idx]
    grads = tape.gradient(label_channel, lclo)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1))
    lclo_shape = lclo[0].shape
    tile = tf.constant([1,lclo_shape[0]],tf.int32)
    grads = tf.tile(pooled_grads[...,tf.newaxis], tile)
    heatcloud = tf.matmul(lclo[0], grads)
    heatcloud = tf.linalg.diag_part(heatcloud)
    heatcloud = tf.squeeze(heatcloud)
    heatcloud = tf.maximum(heatcloud, 0) / tf.math.reduce_max(heatcloud)
    return heatcloud.numpy()

def save_and_display_gradcam(point_cloud, heatcloud, result_path, fileid, i=None, label_names=None):

    pc = point_cloud
    o = np.ones((len(heatcloud),1))
    h = (2/3) * (1 - heatcloud)
    h = np.reshape(h, (len(heatcloud),1))
    hsv = np.hstack((h,o,o))
    rgb = mpl.colors.hsv_to_rgb(hsv)
    # Convert back to open3d pc
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc)
    cloud.colors = o3d.utility.Vector3dVector(rgb)
    #o3d.visualization.draw_geometries([cloud])
    try:
        o3d.io.write_point_cloud(result_path + fileid + "Point_Cloud_Intensity" + label_names[i] + ".ply", cloud)
    except:
        print("cloud not written")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(cloud)
    vis.get_render_option().point_size = 4
    vis.get_render_option().background_color = np.asarray([1, 1, 1])  
    ctr = vis.get_view_control()
    ctr.rotate(180.0, 180.0)  
    vis.update_geometry(cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(result_path + fileid + "Point_Cloud_Intensity" + label_names[i] + ".png")
    vis.destroy_window()
    
database = "AFBMData_NoChairs_Augmented.csv"
train_ds, val_ds, label_weights, val_paths = generate_dataset(filename=database)
pn_model = pointnet(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=False)
pn_model.load_weights('MLCPNBestWeights.h5') #'MLCPN_Validation_New2024-04-01_16_5000_30_Learning Rate_0.00025_Epsilon_1e-07pn_weights_25.h5') #
pn_model.compile(run_eagerly=True)
example_clouds = val_ds.take(BATCH_SIZE)
example_clouds = example_clouds.batch(BATCH_SIZE)
example_paths = val_paths.take(BATCH_SIZE)
points, y_true = list(example_clouds)[0]
y_pred = pn_model.predict(example_clouds, batch_size=BATCH_SIZE)
paths = list(example_paths)
lln = 'conv1d_10' #'dot_1' #'activation_14' #
pn_model.layers[-1].activation = None
result_path = "C:/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/gcam_results/"
for j in range(len(y_pred)):
    print(f"Getting GradCAM for Point Cloud {j}")
    yp = y_pred[j]
    yt = y_true[j]
    p = points[j]
    fileid = np.array2string(paths[j].numpy())
    fileid = fileid.split(".")[0]
    for character in "['":
        fileid = fileid.replace(character, '')
    fileid = fileid[1:]
    pred_label_index = np.where(yp >= 0.5)[0]
    
    for i in pred_label_index:
        heatcloud = gradcam_heatcloud(p, pn_model, lln, label_idx=i)
        #print(heatcloud)
        save_and_display_gradcam(p, heatcloud, result_path, fileid, i, label_names)