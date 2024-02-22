import tensorflow as tf
import numpy as np
import pandas as pd
from model import pointnet, generator
from utils import PerLabelMetric, GarbageMan
from dataset import generate_dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


NUM_POINTS = 5000
NUM_CLASSES = 25
TRAINING = False
LEARN_RATE = 0.0003
BATCH_SIZE = 32
NUM_EPOCHS = 10
username = 'Zachariah'
database = "AFBMData_NoChairs_Augmented.csv"
train_ds, val_ds, label_weights = generate_dataset(filename=database)

pn = pointnet(num_classes=NUM_CLASSES,num_points=NUM_POINTS,train=TRAINING)
pn.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE),
    metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=0.5),
            PerLabelMetric(num_labels=NUM_CLASSES),
            ],      
    run_eagerly=True,
)


tist = pn.fit(x=train_ds, epochs=NUM_EPOCHS, class_weight=label_weights, validation_data=val_ds, callbacks=[GarbageMan()])
