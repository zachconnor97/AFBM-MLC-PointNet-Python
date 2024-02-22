import tensorflow as tf
import numpy as np
import pandas as pd
from model import pointnet, generator
from utils import PerLabelMetric, GarbageMan
from dataset import generate_dataset
import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


NUM_POINTS = 5000
NUM_CLASSES = 25
TRAINING = False
LEARN_RATE = 0.0003
BATCH_SIZE = 32
NUM_EPOCHS = 10
username = 'Zachariah'
database = "AFBMData_NoChairs_Augmented.csv"
save_path = str('/mnt/c/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/' + str(date.today()) + '_' + str(BATCH_SIZE) + '_' + str(NUM_POINTS) + '_' + str(NUM_EPOCHS) + '_' + 'Learning Rate_' + str(LEARN_RATE))


train_ds, val_ds, label_weights = generate_dataset(filename=database)
with open("Label_Weights.csv", mode='w') as f:
    writer = csv.writer(f)
    for key, value in label_weights.items():
        writer.writerow([key, value])

pn = pointnet(num_classes=NUM_CLASSES,num_points=NUM_POINTS,train=TRAINING)
pn.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE),
    metrics=[
            PerLabelMetric(num_labels=NUM_CLASSES),
            tf.keras.metrics.BinaryAccuracy(threshold=0.5),
            tf.keras.metrics.Precision(thresholds=[0.5,1]),
            tf.keras.metrics.Recall(thresholds=[0.5,1]),
            ],      
    run_eagerly=True,
)

tist = pn.fit(x=train_ds, epochs=NUM_EPOCHS, class_weight=label_weights, validation_data=val_ds, callbacks=[GarbageMan()])
## Save history file
histdf = pd.DataFrame(tist.history[0]).T
histfile = save_path + '_train_history_per_label_met.csv'
with open(histfile, mode='w') as f:
    histdf.to_csv(f)

histdf = pd.DataFrame(tist.history[1:])
histfile = save_path + '_train_history.csv'
with open(histfile, mode='w') as f:
    histdf.to_csv(f)

# Validation / Evaluation per Label
data = []
pn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE),
    metrics=[
        PerLabelMetric(num_labels=NUM_CLASSES),
        ],
        run_eagerly=True,
    )
data=pn.evaluate(x=val_ds)
metrics = data[1]
metrics = pd.DataFrame(metrics).T
print(metrics)
histfile = save_path + '_label_validation_allmets_2.csv'

with open(histfile, mode='w') as f:
    metrics.to_csv(f)