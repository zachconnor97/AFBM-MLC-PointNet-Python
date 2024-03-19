import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import date 
import os
import csv
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import pointnet, generator, OrthogonalRegularizer, orthogonal_regularizer_from_config
from utils import PerLabelMetric, GarbageMan
from dataset import generate_dataset
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.src import backend_config
epsilon = backend_config.epsilon
EPS = 1e-7
NUM_POINTS = 5000
NUM_CLASSES = 25
TRAINING = True
LEARN_RATE = 0.25
BATCH_SIZE = 16
NUM_EPOCHS = 18
username = 'Zachariah'
database = "AFBMData_NoChairs_Augmented.csv"
save_path = str('/mnt/c/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/MLCPN_Validation' + str(date.today()) + '_' + str(BATCH_SIZE) + '_' + str(NUM_POINTS) + '_' + str(NUM_EPOCHS) + '_' + 'Learning Rate_' + str(LEARN_RATE) + '_' + 'Epsilon: ' + str(EPS))

label_names = [
    'ConvertCEtoKE,AE,TE', 'ConvertEEtoAE', 'ConvertEEtoLE',
    'ConvertKEtoAE', 'ConvertKEtoEE', 'ConvertLEtoCE',
    'ConvertLEtoEE', 'ExportAE', 'ExportAEtoTE',
    'ExportCE', 'ExportEE', 'ExportGas', 'ExportLE',
    'ExportLiquid', 'ExportSolid', 'ImportEE',
    'ImportGas', 'ImportHE', 'ImportKE',
    'ImportLE', 'ImportLiquid', 'ImportSolid',
    'StoreGas', 'StoreLiquid', 'StoreSolid'
]

g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)
pn_model = pointnet(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=True)
#print(pn_model.get_weights()[0])
#print(pn_model.get_weights()[1])
EStop = EarlyStopping(monitor='val_loss',patience=3, mode='min')
patience = 0
echeck = 0
ediff = 0.001
cur_loss = 0.0
prev_loss = 0.0

def loss(target_y, predicted_y, label_weights=None):
    # Update to binary cross entropy loss
    target_y = tf.cast(target_y, dtype=tf.float32)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss = bce(target_y, predicted_y) 
    return loss

def wbce_loss(target_y, predicted_y, label_weights=None):
    from keras.src import backend, backend_config
    epsilon = backend_config.epsilon
    target = tf.convert_to_tensor(target_y, dtype='float32')
    output = tf.convert_to_tensor(predicted_y, dtype='float32')
    epsilon_ = tf.constant(epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    bceloss = target * tf.math.log(output + epsilon())
    bceloss += (1-target) * tf.math.log(1 - output + epsilon())
    if label_weights != None:
        lw=np.array(list(label_weights.items()))
        lw = lw[:,1]
        wbceloss = backend.mean(-bceloss * lw) 
    else:
        wbceloss = backend.mean(-bceloss) 
    return wbceloss

def train(pn_model, train_ds, label_weights=None): # X is points and Y is labels
    stacked_loss = 0 
    for step, (xbt, ybt) in enumerate(train_ds):
        #print(f"Step: {step}")
        with tf.GradientTape() as t:
            # Trainable variables are automatically tracked by GradientTape
            #current_loss = loss(ybt, pn_model(xbt))
            current_loss = wbce_loss(ybt, pn_model(xbt), label_weights)
            stacked_loss = stacked_loss + current_loss
        #print(f"Current Loss: {current_loss}")
        grads = t.gradient(current_loss, pn_model.trainable_weights)    
        g_optimizer.apply_gradients(zip(grads, pn_model.trainable_weights))
    return stacked_loss/step

def validate(pn_model, val_ds, label_weights): # X is points and Y is labels
    stacked_loss = 0 
    for step, (xbt, ybt) in enumerate(val_ds):
        #print(f"Step: {step}")
        with tf.GradientTape() as t:
            # Trainable variables are automatically tracked by GradientTape
            current_loss = wbce_loss(ybt, pn_model(xbt))
            stacked_loss = stacked_loss + current_loss
        #print(f"Current Loss: {current_loss}")
        #grads = t.gradient(current_loss, pn_model.trainable_weights)    
        # Subtract the gradient scaled by the learning rate
        #g_optimizer.apply_gradients(zip(grads*learn_rate, pn_model.trainable_weights))
    return stacked_loss/step

# Define a training loop
"""
def report(pn_model, loss):
  return f"W = {pn_model.get_weights()[0]:1.2f}, b = {pn_model.get_weights()[1]:1.2f}, loss={loss:2.5f}"
"""

def training_loop(pn_model, train_ds, val_ds, label_weights):
    print(f"Starting:")
    prev_loss = 0
    echeck = 0
    weights = []
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch}:")
        # Update the model with the single giant batch
        e_loss = train(pn_model, train_ds, label_weights=label_weights)
        print(f"Training Loss: {e_loss}")
        # Track this before I update
        weights.append(pn_model.get_weights())
        #print(f"W = {pn_model.get_weights()[0]}, B = {pn_model.get_weights()[1]}")
        # Add weights and biases saving here
        vloss = validate(pn_model, val_ds, label_weights)
        print(f"Validation Loss: {vloss}")
          
        cur_loss = vloss
        prev_loss = cur_loss # PLACEHOLDER
        if abs(prev_loss - cur_loss) < ediff:
            echeck = echeck + 1
            pn_model.save_weights('pn_weights.h5', overwrite=True)
            if echeck > patience:
                pn_model.load_weights('pn_weights_' + epoch-1 + '.h5')
                print("Validation loss not improving. Breaking the training loop.")
                break
        else:
            pn_model.save_weights(str('pn_weights_' + epoch + '.h5'), overwrite=True)
            echeck = 0
        prev_loss = cur_loss

#Callback for saving best model
model_checkpoint = ModelCheckpoint(
    filepath=save_path,
    monitor='val_loss',  # Monitor validation loss
    save_best_only=True,  # Save only the best model
    mode='min',  # Save when validation loss is minimized
    verbose=1  # Show information about saving
)

#train_ds, val_ds, label_weights = generate_dataset(filename=database)

# pn_model Code for the training loop

#current_loss = loss(y = train_points, pn_model(train_label))

#print(f"Starting:")
#print("    ", report(pn_model, current_loss=1))
#training_loop(pn_model, train_ds, val_ds, label_weights)
#pn_model.save(save_path + '_AFBM Model')
"""
# Validation / Evaluation per Label
data = []
pn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE, epsilon=EPS),
    metrics=[
        PerLabelMetric(num_labels=NUM_CLASSES),
        ],
        run_eagerly=True,
    )
data=pn_model.evaluate(x=val_ds)
metrics = data[1]
metrics = pd.DataFrame(metrics).T
#print(metrics)
histfile = save_path + '_label_validation_allmets.csv'

with open(histfile, mode='w') as f:
    metrics.to_csv(f)
"""

def gradcam_heatcloud(cloud, model, lcln, label_idx=None):
    """
    Inputs:
    cloud: Point Cloud Input. Must be tensor shaped (1, Num Points, Num Dims)
    model: Point Net Model. Takes in point cloud, returns labels
    lcln: Name of the last convolutional layer in PointNet
    label_idx: Index of label to generate heatcloud for
    Need to modify function to accept multiple labels and create multiple intensity vectors
    """
    # from keras.io but need to modify for pcs instead
    gradm = tf.keras.models.Model(
        model.inputs, [model.get_layer(lcln).output, model.output]
    )
    with tf.GradientTape() as tape:
        lclo, preds = gradm(cloud)
        #print(preds)
        if label_idx is None:
            label_idx = tf.argmax(preds[0])
        label_channel = preds[:, label_idx]
    #print(label_idx)
    grads = tape.gradient(label_channel, lclo)
    #print(grads)
    #pooled_grads = tf.reduce_mean(grads, axis=0) # Dimensionality of this var is causing issue in line 190...
    pooled_grads = tf.reduce_mean(grads, axis=(0,1)) 
    #print(pooled_grads[..., tf.newaxis])
    lclo = lclo[0]

    #Checking the shape of the matrices before multipication
    lclo_shape = lclo.shape
    pooled_grads = pooled_grads[..., tf.newaxis]
    #print("Shape of lclo:", lclo_shape)
    #print("Shape of pooled_grads:", pooled_grads.shape)

    # Perform matrix multiplication
    heatcloud = lclo @ pooled_grads
    #heatcloud = lclo @ pooled_grads[..., tf.newaxis] #error here.
    #print(heatcloud.shape)
    heatcloud = tf.squeeze(heatcloud)
    heatcloud = tf.maximum(heatcloud, 0) / tf.math.reduce_max(heatcloud)
    return heatcloud.numpy()

def save_and_display_gradcam(point_cloud, heatcloud, i=None, label_names=None):
    
    pc = point_cloud
    v = np.zeros((len(heatcloud),1))
    #rg[:,0] = np.subtract(rg[:,0], (1 + np.log(heatcloud)))
    #rg[:,1] = np.subtract(rg[:,1], (1 + np.log(heatcloud)))
    #b = np.ones((len(heatcloud),1))

    #g = 1 + np.log(heatcloud)
    g = heatcloud
    g = np.reshape(g, (len(heatcloud),1))
    rgb = np.hstack((v,g,v))
    # Convert back to open3d pc
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc)
    cloud.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([cloud])

    #o3d.io.write_point_cloud("Point_Cloud_Intensity" + label_names[i] + ".ply", cloud)

# Test GradCAM stuff
pn_model.load_weights('MLCPNBestWeights.h5')
#pn_model.summary()
#testcloud = o3d.io.read_point_cloud('C:/Users/gabri/OneDrive - Oregon State University/AllClouds10k/AllClouds10k/lamp_3636649_be13324c84d2a9d72b151d8b52c53b901_10000_2pc.ply') # use open3d to import point cloud from file
#testcloud = o3d.io.read_point_cloud('C:/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AllClouds10k/AllClouds10k/bottle_2876657_2618100a5821a4d847df6165146d5bbd1_10000_2pc.ply') # use open3d to import point cloud from file
#testcloud = o3d.io.read_point_cloud('/mnt/c/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AllClouds10k/AllClouds10k/lamp_3636649_be13324c84d2a9d72b151d8b52c53b901_10000_2pc.ply') # use open3d to import point cloud from file
pc_path = 'C:/Users/gabri/OneDrive - Oregon State University/AllClouds10k/AllClouds10k/sofa_couch_lounge_4256520_3e3ad2629c9ab938c2eaaa1f79e71ec1_10000_2pc.ply'
#pc_path = 'C:/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AllClouds10k/AllClouds10k/sofa_couch_lounge_4256520_3e3ad2629c9ab938c2eaaa1f79e71ec1_10000_2pc.ply'
#bottle_2876657_2618100a5821a4d847df6165146d5bbd1_10000_2pc.ply'
#lamp_3636649_be13324c84d2a9d72b151d8b52c53b901_10000_2pc.ply'
pc = o3d.io.read_point_cloud(pc_path)
pc = pc.uniform_down_sample(every_k_points=2)
pc= pc.points
pc = np.asarray([pc])[0]
testcloud = np.reshape(pc, (1,5000,3))
testcloud = tf.constant(testcloud, dtype='float64')

lln = 'dot' #'activation_14'
labels = pn_model.predict(testcloud)
print("Predicted Labels: ", labels)
pn_model.layers[-1].activation = None

# Get idx from predicted labels with prediction > 0.5 
labels = np.array(labels)
label_index = np.where(labels >= 0.5)[1]
#print(label_index)

for i in label_index:
    heatcloud = gradcam_heatcloud(testcloud, pn_model, lln, label_idx=i)
    #print(heatcloud)
    save_and_display_gradcam(pc, heatcloud, i, label_names)