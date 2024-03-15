
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
NUM_POINTS = 5000
NUM_CLASSES = 25
TRAINING = False
BATCH_SIZE = 100

from model import pointnet, OrthogonalRegularizer, orthogonal_regularizer_from_config
from dataset_example import generate_dataset

username = 'Zachariah'
database = "AFBMData_NoChairs_Augmented.csv"
save_path = str('/mnt/c/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/MLCPN_Validation_Examples' + str(date.today()) + '_' + str(BATCH_SIZE) + '_' + str(NUM_POINTS))
pn_model = pointnet(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=TRAINING)
pn_model.load_weights('weights/pn_weights_28.h5')
train_ds, val_ds, label_weights, val_paths = generate_dataset(filename=database)

example_clouds = val_ds.take(BATCH_SIZE)
example_clouds = example_clouds.batch(BATCH_SIZE)
example_paths = val_paths.take(BATCH_SIZE)
points, y_true = list(example_clouds)[0]
#print(f"Labels: {labels}")
pn_model.compile(run_eagerly=True)
y_pred = pn_model.predict(example_clouds, batch_size=BATCH_SIZE)
#labels = [y_true.numpy(), y_pred]
#print(labels)

files = pd.DataFrame(example_paths)
file = save_path + 'File Paths.csv'
with open(file, mode='w') as f:
    files.to_csv(f)

labels = pd.DataFrame(y_true.numpy())
file = save_path + 'True Labels.csv'
with open(file, mode='w') as f:
    labels.to_csv(f)

labels = pd.DataFrame(y_pred)
file = save_path + 'Pred Labels.csv'
with open(file, mode='w') as f:
    labels.to_csv(f)
