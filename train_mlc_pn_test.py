import tensorflow as tf
import numpy as np
import pandas as pd
from model import pointnet, generator
from AFBM_MLC import PerLabelMetric, generate_dataset, GarbageMan
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUM_POINTS = 5000
NUM_CLASSES = 25
TRAINING = True
LEARN_RATE = 0.0003
BATCH_SIZE = 32
NUM_EPOCHS = 1

database = "AFBMData_NoChairs_Augmented.csv"
train_ds, val_ds, label_weights = generate_dataset(filename=database)

pn = pointnet(num_classes=NUM_CLASSES,num_points=NUM_POINTS,training=TRAINING)
pn.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE),
    metrics=[
            PerLabelMetric(num_labels=NUM_CLASSES),
            tf.keras.metrics.BinaryAccuracy(threshold=0.5),
            tf.keras.metrics.Precision(thresholds=[0.5,1]),
            tf.keras.metrics.Recall(thresholds=[0.5,1]),
            tf.keras.metrics.F1Score(threshold=0.5),
            ],      
    run_eagerly=True,
)

gen = generator(num_points=NUM_POINTS,num_classes=NUM_CLASSES,training=TRAINING)
gen.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE),
    metrics=[
        PerLabelMetric(num_labels=NUM_CLASSES),
        tf.keras.metrics.BinaryAccuracy(threshold=0.5),
        tf.keras.metrics.Precision(thresholds=[0.5,1]),
        tf.keras.metrics.Recall(thresholds=[0.5,1]),
            tf.keras.metrics.F1Score(threshold=0.5),
    ],
    run_eagerly=True,
)

#thist = pn.fit(x=train_ds, epochs=NUM_EPOCHS, class_weight=label_weights, validation_data=val_ds, callbacks=[GarbageMan()])

#val_data = pn.evaluate(x=val_ds)