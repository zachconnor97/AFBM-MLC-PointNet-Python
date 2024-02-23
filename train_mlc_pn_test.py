import tensorflow as tf
import numpy as np
import pandas as pd
from model import pointnet, generator
from utils import PerLabelMetric, GarbageMan
from dataset import generate_dataset, generator_dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


NUM_POINTS = 2000
NUM_CLASSES = 25
TRAINING = False
LEARN_RATE = 0.0003
BATCH_SIZE = 32
NUM_EPOCHS = 10
username = 'Zachariah'
database = "AFBMData_NoChairs_Augmented.csv"
"""
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
"""
gen = generator(num_points=NUM_POINTS,num_classes=NUM_CLASSES,train=TRAINING)
gen.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE),
    metrics=[
        #PerLabelMetric(num_labels=NUM_CLASSES),
        tf.keras.metrics.BinaryAccuracy(threshold=0.5),
        #tf.keras.metrics.Precision(thresholds=[0.5,1]),
        #tf.keras.metrics.Recall(thresholds=[0.5,1]),
        #tf.keras.metrics.F1Score(threshold=0.5),
    ],
    run_eagerly=True,
)

#t_dum = tf.constant(np.random.rand(BATCH_SIZE,NUM_POINTS,3),dtype='float64')
#l_dum = tf.constant(np.round(np.random.rand(BATCH_SIZE,NUM_CLASSES)),dtype='float64')
#print(np.shape(t_dum))
#print(np.shape(l_dum))
#thist = pn.fit(x=train_ds, epochs=NUM_EPOCHS, class_weight=label_weights, validation_data=val_ds, callbacks=[GarbageMan()])

#val_data = pn.evaluate(x=t_dum, y=l_dum,batch_size=BATCH_SIZE)
#print(pn.output_shape)

#print(gen.output_shape)
#gen_pred = gen.predict(x=l_dum, batch_size=BATCH_SIZE) #gen.evaluate(x=l_dum, y=t_dum,batch_size=BATCH_SIZE)
gtrain_ds, gval_ds, label_weights = generator_dataset(filename=database)
gentrain = gen.fit(x=gtrain_ds,validation_data=gval_ds, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
#print(np.shape(gen_pred))
#print("\t Generator Output: %d",gen_pred)
#gentrain = gen.evaluate(x=val_ds, batch_size=BATCH_SIZE)
#print(gentrain)