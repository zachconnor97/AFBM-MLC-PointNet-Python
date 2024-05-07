import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import open3d as o3d
import matplotlib as mpl
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt
from model import pointnet
from keras.src import backend_config
epsilon = backend_config.epsilon
NUM_CLASSES = 25
TRAINING = False

database = "AFBMData_NoChairs_Augmented.csv"
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
    heatcloud = heatcloud / (tf.maximum(tf.math.reduce_max(tf.math.abs(heatcloud)), 0)) 
    return heatcloud.numpy()

def display_gradcam(point_cloud, heatcloud):
    pc = point_cloud
    o = np.ones((len(heatcloud),1))
    h = (2/3) * (1 - np.abs(heatcloud)) 
    h = np.reshape(h, (len(heatcloud),1))
    hsv = np.hstack((h,o,o))
    rgb = mpl.colors.hsv_to_rgb(hsv)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc)
    cloud.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([cloud])

pc_path =  'bottle_2876657_225d046d8f798afacba9caf4d254cef01_10000_2pc.ply'
pc = o3d.io.read_point_cloud(pc_path)
pc = pc.points
pc = np.asarray([pc])[0]
NUM_POINTS = len(pc)
testcloud = np.reshape(pc, (1,NUM_POINTS,3))
testcloud = tf.constant(testcloud, dtype='float64')

pn_model = pointnet(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=False)
pn_model.load_weights('MLCPNBestWeights.h5')
pn_model.compile(run_eagerly=True)
lln = 'activation_14'
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
pn_model.layers[-1].activation = None

# Get idx from predicted labels with prediction > 0.5 
y_pred = np.array(y_pred)
pred_label_index = np.where(y_pred >= 0.5)[1]

for i in pred_label_index:
    heatcloud = gradcam_heatcloud(testcloud, pn_model, lln, label_idx=i)
    display_gradcam(pc, heatcloud)

