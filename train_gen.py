import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import date 
import os
import csv
import open3d as o3d
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
NUM_EPOCHS = 1
username = 'Zachariah'
database = "AFBMData_NoChairs_Augmented.csv"
save_path = str('/mnt/c/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/' + 'Generator' + str(date.today()) + '_' + str(BATCH_SIZE) + '_' + str(NUM_POINTS) + '_' + str(NUM_EPOCHS) + '_' + 'Learning Rate_' + str(LEARN_RATE) + '_' + 'Epsilon: ' + str(EPS))

g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)
gmodel = generator(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=True)
gmodel.compile(run_eagerly=True)
EStop = EarlyStopping(monitor='val_loss',patience=3, mode='min')

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

def CD_loss(tt, tg): # Chamfer Distance Loss Function

  def distance_matrix(array1, array2):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
            , it's size: (num_point, num_point)
    """
    num_point, num_features = array1.shape
    expanded_array1 = tf.tile(array1, (num_point, 1))
    expanded_array2 = tf.reshape(
            tf.tile(tf.expand_dims(array2, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = tf.norm(expanded_array1-expanded_array2, axis=1)
    distances = tf.reshape(distances, (num_point, num_point))
    return distances

  def av_dist(array1, array2):
    """
    arguments:
        array1, array2: both size: (num_points, num_feature)
    returns:
        distances: size: (1,)
    """
    distances = distance_matrix(array1, array2)
    distances = tf.reduce_min(distances, axis=1)
    distances = tf.reduce_mean(distances)
    return distances

  def av_dist_sum(arrays):
    """
    arguments:
        arrays: array1, array2
    returns:
        sum of av_dist(array1, array2) and av_dist(array2, array1)
    """
    tt, tg = arrays
    av_dist1 = av_dist(tt, tg)
    av_dist2 = av_dist(tt, tg)
    return av_dist1+av_dist2

  def chamfer_distance_tf(tt, tg):
      batch_size, num_point, num_features = tt.shape
      dist = tf.reduce_mean(
        tf.map_fn(av_dist_sum, elems=(tt, tg), dtype=tf.float64)
           )
      return dist
      
  dist_tf = chamfer_distance_tf(tt, tg)
  return dist_tf
  
def train(gmodel, train_ds, LEARN_RATE): # X is labels and Y is train_ds
  stacked_loss = 0 
  for step, (xbt, ybt) in enumerate(train_ds):
    with tf.GradientTape(persistent=True) as t:
      t.watch(xbt)
      # Trainable variables are automatically tracked by GradientTape
      pred = gmodel(xbt)
      current_loss = CD_loss(ybt, pred) 
      
      stacked_loss = stacked_loss + current_loss
    print(f"Step: {step}, CD Loss: {current_loss}")
    grads = t.gradient(current_loss, gmodel.trainable_weights)  
    if grads == None:
      print("No Gradients")
    else:
      try:
        g_optimizer.apply_gradients(zip(grads, gmodel.trainable_weights))
      except:
        print("Gradients Not Applied")
  return stacked_loss/step

def training_loop(gmodel, train_ds):
  for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch}:")
    # Update the model with the single giant batch
    e_loss = train(gmodel, train_ds, LEARN_RATE=0.001)
    print(f"Mean Loss: {e_loss}")
    gmodel.save_weights(str(save_path + 'pn_weights_' + str(epoch) + '.h5'), overwrite=True)


train_ds, val_ds, label_weights, train_label, train_points, val_label, val_points, val_paths = generator_dataset(filename=database)

# gmodel Code for the training loop

print(f"Starting:")
training_loop(gmodel, train_ds)


#gmodel = generator(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=False)
#gmodel.compile(run_eagerly=True)
#gmodel.load_weights("C:/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/Generator2024-04-04_16_2000_3_Learning Rate_2.5e-06_Epsilon_1e-07pn_weights_0.h5")
"""
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
"""