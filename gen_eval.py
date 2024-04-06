
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import date 
import open3d as o3d
from model import pointnet, generator, OrthogonalRegularizer, orthogonal_regularizer_from_config
from utils import PerLabelMetric, GarbageMan
from dataset import generator_dataset
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.src import backend
EPS = 1e-7
NUM_POINTS = 200
NUM_CLASSES = 25
TRAINING = True
LEARN_RATE = 0.0000025
BATCH_SIZE = 8
NUM_EPOCHS = 15
username = 'Zachariah'
database = "AFBMData_NoChairs_Augmented.csv"
save_path = str('/mnt/c/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/' + 'Generator_pcLoss' + str(date.today()) + '_' + str(BATCH_SIZE) + '_' + str(NUM_POINTS) + '_' + str(NUM_EPOCHS) + '_' + 'Learning Rate_' + str(LEARN_RATE) + '_' + 'Epsilon: ' + str(EPS))

g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)
gmodel = generator(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=True)
gmodel.compile(run_eagerly=True)
EStop = EarlyStopping(monitor='val_loss',patience=3, mode='min')

gmodel = generator(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=False)
gmodel.compile(run_eagerly=True)
gmodel.load_weights("C:/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/Generator_CDLoss2024-04-05_8_200_15_Learning Rate_2.5e-06_Epsilon_1e-07gen_weights_11.h5")
train_ds, val_ds, label_weights, train_label, train_points, val_label, val_points, val_paths = generator_dataset(filename=database)

BATCH_SIZE = 1
examples = val_ds.take(1)
#print(examples)
#examples = examples.batch(1)
#examples = examples.batch(1)
example_paths = val_paths.take(1)
labels, points = list(examples)[0]
c_gen = gmodel.predict(examples, batch_size=1)


tcloud = o3d.geometry.PointCloud()
tcloud.points = o3d.utility.Vector3dVector(points[0].numpy())

gcloud = o3d.geometry.PointCloud()
gcloud.points = o3d.utility.Vector3dVector(c_gen[0])
o3d.visualization.draw_geometries([gcloud])
