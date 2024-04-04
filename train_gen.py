import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import date 
import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import pointnet, generator, OrthogonalRegularizer, orthogonal_regularizer_from_config
from utils import PerLabelMetric, GarbageMan
from dataset import generator_dataset
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.src import backend
EPS = 1e-7
NUM_POINTS = 2000
NUM_CLASSES = 25
TRAINING = True
LEARN_RATE = 0.0000025
BATCH_SIZE = 16
NUM_EPOCHS = 5
username = 'Zachariah'
database = "AFBMData_NoChairs_Augmented.csv"
save_path = str('/mnt/c/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/' + 'Generator' + str(date.today()) + '_' + str(BATCH_SIZE) + '_' + str(NUM_POINTS) + '_' + str(NUM_EPOCHS) + '_' + 'Learning Rate_' + str(LEARN_RATE) + '_' + 'Epsilon: ' + str(EPS))

g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)
gmodel = generator(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=True)
gmodel.compile(run_eagerly=True)
EStop = EarlyStopping(monitor='val_loss',patience=3, mode='min')

def loss(target_y, predicted_y, label_weights=None):
    # Update to binary cross entropy loss
    target_y = tf.cast(target_y, dtype=tf.float32)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss = bce(target_y, predicted_y) 
    return loss

def pc_loss(tt, tg):
  tt = tf.cast(tt, dtype=tf.float64)
  tg = tf.cast(tg, dtype=tf.float64)
  # find average of x, y, and z coords
  xt_mn = backend.mean(tt[:,0])
  yt_mn = backend.mean(tt[:,1])
  zt_mn = backend.mean(tt[:,2])
  xg_mn = backend.mean(tg[:,0])
  yg_mn = backend.mean(tg[:,1])
  zg_mn = backend.mean(tg[:,2])

  # find second moment of inertia tensors M2_g M2_t
  tixx = backend.mean(tf.math.subtract(tt[:,0],xt_mn) ** 2)
  tiyy = backend.mean(tf.math.subtract(tt[:,0],yt_mn) ** 2)
  tizz = backend.mean(tf.math.subtract(tt[:,0],zt_mn) ** 2)
  tixy = backend.mean(tf.matmul(tf.math.subtract(tt[:,0],xt_mn), tf.transpose(tf.math.subtract(tt[:,0],yt_mn))))
  tixz = backend.mean(tf.matmul(tf.math.subtract(tt[:,0],xt_mn), tf.transpose(tf.math.subtract(tt[:,0],zt_mn))))
  tiyz = backend.mean(tf.matmul(tf.math.subtract(tt[:,0],yt_mn), tf.transpose(tf.math.subtract(tt[:,0],zt_mn))))
  M2_t = tf.stack([[tixx, tixy, tixz],
                  [tixy, tiyy, tiyz],
                  [tixz, tiyz, tizz]])
  gixx = backend.mean(tf.math.subtract(tg[:,0],xg_mn) ** 2) 
  giyy = backend.mean(tf.math.subtract(tg[:,0],yg_mn) ** 2)
  gizz = backend.mean(tf.math.subtract(tg[:,0],zg_mn) ** 2)
  gixy = backend.mean(tf.matmul(tf.math.subtract(tg[:,0],xg_mn), tf.transpose(tf.math.subtract(tg[:,0],yg_mn))))
  gixz = backend.mean(tf.matmul(tf.math.subtract(tg[:,0],xg_mn), tf.transpose(tf.math.subtract(tg[:,0],zg_mn))))
  giyz = backend.mean(tf.matmul(tf.math.subtract(tg[:,0],yg_mn), tf.transpose(tf.math.subtract(tg[:,0],zg_mn))))
  M2_g = tf.stack([[gixx, gixy, gixz],
                  [gixy, giyy, giyz],
                  [gixz, giyz, gizz]])
  M2_t_inv = tf.linalg.inv((M2_t))
  r_inv = tf.matmul(M2_g,tf.transpose(M2_t_inv))
  r = tf.linalg.inv(r_inv)
  eye = tf.linalg.eye(3,3)
  r = tf.cast(r, dtype=tf.float32)
  pc_loss = backend.abs(r - eye)
  pc_loss = backend.mean(pc_loss)
  #print(f"PointCloud Loss [unnormalized] = {pc_loss}")
  pc_loss = tf.math.add(1.0, tf.math.log(pc_loss)) 
  pc_loss = tf.math.abs(pc_loss)
  #pc_loss = 2
  return pc_loss

def train(gmodel, train_ds, LEARN_RATE): # X is labels and Y is train_ds
  stacked_loss = 0 
  for step, (xbt, ybt) in enumerate(train_ds):
    with tf.GradientTape(persistent=True) as t:
      t.watch(xbt)
      # Trainable variables are automatically tracked by GradientTape
      pred = gmodel(xbt)
      current_loss = pc_loss(ybt, pred) 
      
      stacked_loss = stacked_loss + current_loss
    grads = t.gradient(current_loss, gmodel.trainable_weights)  
    if grads == None:
      print("No Gradients")
    else:
      try:
        g_optimizer.apply_gradients(zip(grads, gmodel.trainable_weights))
      except:
        print("Gradients Not Applied")
  return stacked_loss/step

# Define a training loop
"""
def report(gmodel, loss):
  return f"W = {gmodel.get_weights()[0]:1.2f}, b = {gmodel.get_weights()[1]:1.2f}, loss={loss:2.5f}"
"""

def training_loop(gmodel, train_ds):
  for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch}:")
    # Update the model with the single giant batch
    e_loss = train(gmodel, train_ds, LEARN_RATE=0.001)
    print(f"Mean Loss: {e_loss}")
    # Track this before I update
    weights.append(gmodel.get_weights()[0])
    biases.append(gmodel.get_weights()[1])
    #print(f"W = {gmodel.get_weights()[0]}, B = = {gmodel.get_weights()[1]}")

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
