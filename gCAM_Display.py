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
import colorsys
import matplotlib
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
BATCH_SIZE = 100
NUM_EPOCHS = 18
username = 'Zachariah'
database = "AFBMData_NoChairs_Augmented.csv"
save_path = str('C:/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/MLCPN_Validation' + str(date.today()) + '_' + str(BATCH_SIZE) + '_' + str(NUM_POINTS) + '_' + str(NUM_EPOCHS) + '_' + 'Learning Rate_' + str(LEARN_RATE) + '_' + 'Epsilon: ' + str(EPS))

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
    heatcloud = heatcloud / (tf.maximum(tf.math.reduce_max(tf.math.abs(heatcloud)), 0)) #tf.minimum(heatcloud, 0) / tf.math.reduce_max(heatcloud)
    return heatcloud.numpy()

def save_and_display_gradcam(point_cloud, heatcloud, result_path, fileid, i=None, label_names=None):

    pc = point_cloud
    o = np.ones((len(heatcloud),1))
    h = (2/3) * (1 - np.abs(heatcloud)) 
    h = np.reshape(h, (len(heatcloud),1))
    hsv = np.hstack((h,o,o))
    rgb = mpl.colors.hsv_to_rgb(hsv)
    # Convert back to open3d pc
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc)
    cloud.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([cloud])

pc_path = "C:/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AllClouds10k/AllClouds10k/bottle_2876657_225d046d8f798afacba9caf4d254cef01_10000_2pc.ply" 
#"C:/Users/Zachariah Connor/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/gcam_results/RGB conv1d_10 e28 2nd 500/bench_2828884_459f90aa2162b1f1d46c340938e2ff1c1_10000_2pcPoint_Cloud_IntensityImportSolid.ply"
#"C:/Users/Zachariah Connor/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/gcam_results/RGB conv1d_10 e28 2nd 500/guitar_3467517_950b02a2db950424c4a359bec2c174271_10000_2pcPoint_Cloud_IntensityConvertKEtoEE.ply"
#'C:/Users/Zachariah Connor/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/gcam_results/RGB conv1d_10 e28 2nd 500/ashcan_trash can_garbage can_wastebin_ash bin_ash-bin_ashbin_dustbin_trash barrel_trash bin_2747177_3c03342f421e5eb2ad5067eac75a07f71_10000_2pcPoint_Cloud_IntensityImportLiquid.PLY' 
#'C:/Users/Zachariah Connor/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AllClouds10k/AllClouds10k/lamp_3636649_ca968e46ba74732551970742dd5663211_10000_2pc.ply' 
pc = o3d.io.read_point_cloud(pc_path)
#pc = pc.uniform_down_sample(every_k_points=2)
pc = pc.points
pc = np.asarray([pc])[0]
NUM_POINTS = len(pc)
testcloud = np.reshape(pc, (1,NUM_POINTS,3))
testcloud = tf.constant(testcloud, dtype='float64')

pn_model = pointnet(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=False)
pn_model.load_weights('MLCPNBestWeights.h5') #'MLCPN_Validation_New2024-04-01_16_5000_30_Learning Rate_0.00025_Epsilon_1e-07pn_weights_25.h5') #
pn_model.compile(run_eagerly=True)
lln = 'activation_14'#'conv1d_10' #'dense_7' #'dot_1'
y_pred = pn_model.predict(testcloud)
label_names = np.array(label_names)
y_pred1 = y_pred.tolist()[0]
label_names = label_names.tolist()
output = label_names
print(f"Prdicted Labels: \n")
for i in range(0, len(output)):
    output[i] = output[i] + ": " + str(round(y_pred1[i], 5))
    if y_pred1[i] >= 0.5:
        print(f"Label {i}: {output[i]}")
#label_dict = pd.concat(pd.DataFrame(label_names),pd.DataFrame(y_pred))
pn_model.layers[-1].activation = None

# Get idx from predicted labels with prediction > 0.5 
y_pred = np.array(y_pred)
pred_label_index = np.where(y_pred >= 0.5)[1]
print(pred_label_index)

for i in pred_label_index:
    heatcloud = gradcam_heatcloud(testcloud, pn_model, lln, label_idx=i)
    save_and_display_gradcam(pc, heatcloud, i, label_names)

