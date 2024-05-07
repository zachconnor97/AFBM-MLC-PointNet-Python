import tensorflow as tf
import numpy as np
from datetime import date 
import os
import open3d as o3d
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import pointnet
from keras.src import backend_config
epsilon = backend_config.epsilon
NUM_CLASSES = 25
username = 'Zachariah'
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

# File path of point cloud to predict
pc_path = 'bottle_2876657_225d046d8f798afacba9caf4d254cef01_10000_2pc.ply'
pc = o3d.io.read_point_cloud(pc_path)
o3d.visualization.draw_geometries([pc])
pc = pc.points
pc = np.asarray([pc])[0]
NUM_POINTS = len(pc)
testcloud = np.reshape(pc, (1,NUM_POINTS,3))
testcloud = tf.constant(testcloud, dtype='float64')
pn_model = pointnet(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=False)
pn_model.load_weights('MLCPNBestWeights.h5')
pn_model.compile(run_eagerly=True)
y_pred = pn_model.predict(testcloud)
label_names = np.array(label_names)
y_pred1 = y_pred.tolist()[0]
label_names = label_names.tolist()
output = label_names
print(f"Prdicted Labels: \n")
for i in range(0, len(output)):
    output[i] = output[i] + ": " + str(round(y_pred1[i], 5))
    if y_pred1[i] >= 0.0:
        print(f"Label {i}: {output[i]}")