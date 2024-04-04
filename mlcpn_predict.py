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

pc_path = "C:/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/gcam_results/slicing study/lamp_3636649_ca968e46ba74732551970742dd5663211_10000_2pcPoint_Cloud_IntensityConvertEEtoLE.ply"

col_t = 0.25 #0.75 #
pc = o3d.io.read_point_cloud(pc_path)
col = np.asarray([pc.colors])[0]
pidx = np.where(col[:,1] > col_t)[0]
pc = pc.points
pc = np.asarray([pc])[0]
pc = pc[pidx]
col = col[pidx]
NUM_POINTS = len(pc)
testcloud = np.reshape(pc, (1,NUM_POINTS,3))
testcloud = tf.constant(testcloud, dtype='float64')

pn_model = pointnet(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=False)
pn_model.load_weights('MLCPNBestWeights.h5') #'MLCPN_Validation_New2024-04-01_16_5000_30_Learning Rate_0.00025_Epsilon_1e-07pn_weights_25.h5') #
pn_model.compile(run_eagerly=True)
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

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(pc)
cloud.colors = o3d.utility.Vector3dVector(col)
o3d.visualization.draw_geometries([cloud])