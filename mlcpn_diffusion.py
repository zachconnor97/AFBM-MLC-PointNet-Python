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
save_path = str('/mnt/c/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/' + 'Generator_CDLoss' + str(date.today()) + '_' + str(BATCH_SIZE) + '_' + str(NUM_POINTS) + '_' + str(NUM_EPOCHS) + '_' + 'Learning Rate_' + str(LEARN_RATE) + '_' + 'Epsilon: ' + str(EPS))

g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)
gmodel = generator(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=True)
gmodel.compile(run_eagerly=True)
EStop = EarlyStopping(monitor='val_loss',patience=3, mode='min')