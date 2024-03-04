import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import date 
from model import pointnet, generator, OrthogonalRegularizer, orthogonal_regularizer_from_config
from utils import PerLabelMetric, GarbageMan
from dataset import generator_dataset
import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.callbacks import ModelCheckpoint, EarlyStopping

EPS = 1e-7
NUM_POINTS = 2000
NUM_CLASSES = 25
TRAINING = True
LEARN_RATE = 0.000025
BATCH_SIZE = 16
NUM_EPOCHS = 25
username = 'Zachariah'
database = "AFBMData_NoChairs_Augmented.csv"
save_path = str('/mnt/c/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/' + str(date.today()) + '_' + str(BATCH_SIZE) + '_' + str(NUM_POINTS) + '_' + str(NUM_EPOCHS) + '_' + 'Learning Rate_' + str(LEARN_RATE) + '_' + 'Epsilon: ' + str(EPS))

g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)
gmodel = generator(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=True)
print(gmodel.get_weights()[0])
print(gmodel.get_weights()[1])
EStop = EarlyStopping(monitor='val_loss',patience=3, mode='min')

# This computes a single loss value for an entire batch - Zach change this to desired loss function
def loss(target_y, predicted_y):
  target_y = tf.cast(target_y, dtype=tf.float32)  # Assuming float32 is the desired data type
  #print("Target shape:", target_y.shape)
  #print("Predicted shape:", predicted_y.shape)
  return tf.reduce_mean(tf.square(target_y - predicted_y))

def train(gmodel, train_ds, LEARN_RATE): # X is labels and Y is train_ds
  stacked_loss = 0 
  for step, (xbt, ybt) in enumerate(train_ds):
  
    with tf.GradientTape() as t:
      # Trainable variables are automatically tracked by GradientTape
      current_loss = loss(ybt, gmodel(xbt))
      stacked_loss = stacked_loss + current_loss

    grads = t.gradient(current_loss, gmodel.trainable_weights)
    
    
    # Subtract the gradient scaled by the learning rate
    g_optimizer.apply_gradients(zip(grads, gmodel.trainable_weights))
  return stacked_loss/step

# Define a training loop
def report(gmodel, loss):
  return f"W = {gmodel.get_weights()[1].numpy():1.2f}, b = {gmodel.get_weights()[1].numpy():1.2f}, loss={loss:2.5f}"


def training_loop(gmodel, train_ds):

  for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch}:")
    # Update the model with the single giant batch
    e_loss = train(gmodel, train_ds, LEARN_RATE=0.1)

    # Track this before I update
    weights.append(gmodel.get_weights()[0].numpy())
    biases.append(gmodel.get_weights()[0].numpy())
    
    print(f"{report(gmodel, e_loss)}")


#Callback for saving best model
model_checkpoint = ModelCheckpoint(
    filepath=save_path,
    monitor='val_loss',  # Monitor validation loss
    save_best_only=True,  # Save only the best model
    mode='min',  # Save when validation loss is minimized
    verbose=1  # Show information about saving
)

train_ds, val_ds, label_weights, train_label, train_points, val_label, val_points = generator_dataset(filename=database)

"""with open("Label_Weights.csv", mode='w') as f:
    writer = csv.writer(f)
    for key, value in label_weights.items():
        writer.writerow([key, value])

pn = pointnet(num_classes=NUM_CLASSES,num_points=NUM_POINTS,train=TRAINING)
pn.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE, epsilon=EPS),
    metrics=[
            #PerLabelMetric(num_labels=NUM_CLASSES),
            tf.keras.metrics.BinaryAccuracy(threshold=0.5),
            tf.keras.metrics.Precision(thresholds=[0.5,1]),
            tf.keras.metrics.Recall(thresholds=[0.5,1]),
            ],      
    run_eagerly=True,
)
"""

# gmodel Code for the training loop
weights = []
biases = []


#current_loss = loss(y = train_points, gmodel(train_label))

print(f"Starting:")
#print("    ", report(gmodel, current_loss=1))
training_loop(gmodel, train_ds)

"""
tist = pn.fit(x=train_ds, epochs=NUM_EPOCHS, class_weight=label_weights, validation_data=val_ds, callbacks=[GarbageMan(), model_checkpoint, EStop])
pn.save(save_path + '_AFBM Model')

## Save history file
histdf = pd.DataFrame(tist.history)
histfile = save_path + '_train_history_per_label_met.csv'
with open(histfile, mode='w') as f:
    histdf.to_csv(f)


tf.keras.utils.get_custom_objects()['OrthogonalRegularizer'] = OrthogonalRegularizer
pn = tf.keras.models.load_model(save_path + '_AFBM Model', custom_objects={'OrthogonalRegularizer': orthogonal_regularizer_from_config})  #save_path + '_AFBM Model', custom_objects={'OrthogonalRegularizer': orthogonal_regularizer_from_config})


# Validation / Evaluation per Label
data = []
pn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE, epsilon=EPS),
    metrics=[
        PerLabelMetric(num_labels=NUM_CLASSES),
        ],
        run_eagerly=True,
    )
data=pn.evaluate(x=val_ds)
metrics = data[1]
metrics = pd.DataFrame(metrics).T
#print(metrics)
histfile = save_path + '_label_validation_allmets_2.csv'

with open(histfile, mode='w') as f:
    metrics.to_csv(f)

"""