import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import date 
import os
import csv
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
LEARN_RATE = 0.00025
BATCH_SIZE = 16
NUM_EPOCHS = 30
username = 'Zachariah'
database = "AFBMData_NoChairs_Augmented.csv"
save_path = str('/mnt/c/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/MLCPN_Validation_MOD_WEIGHTS' + str(date.today()) + '_' + str(BATCH_SIZE) + '_' + str(NUM_POINTS) + '_' + str(NUM_EPOCHS) + '_Learning Rate_' + str(LEARN_RATE) + '_Epsilon_' + str(EPS))

g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)
pn_model = pointnet(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=False)
#print(pn_model.get_weights()[0])
#print(pn_model.get_weights()[1])
patience = 5
echeck = 0
ediff = 0.0025
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
        #prev_loss = cur_loss # PLACEHOLDER
        pn_model.save_weights(str(save_path + 'pn_weights_' + str(epoch) + '.h5'), overwrite=True)
        if abs(prev_loss - cur_loss) < ediff:
            echeck = echeck + 1
            if echeck > patience:
                try:
                    pn_model.load_weights(save_path + 'pn_weights_' + str(epoch-echeck) + '.h5')
                    print(f"Validation loss not improving, \nLoaded best weights")
                except:
                    print(f"Validation loss not improving. \nUnable to load weights, using last weights")
                break
        else:
            echeck = 0
        prev_loss = cur_loss
    try:
        pn_model.load_weights(save_path + 'pn_weights_' + str(epoch-echeck) + '.h5')
        print(f"Weights from Epoch {epoch-echeck} Loaded \nWeight Load #2")
    except:
        print("Unable to load weights, using last weights")

train_ds, val_ds, label_weights = generate_dataset(filename=database)
print(f"Label Weights: {label_weights}")
#label_weights[11] = 10
#label_weights[16] = 10
#print(f"Adjusted Label Weights: {label_weights}")

#training_loop(pn_model, train_ds, val_ds, label_weights)

#pn_model.save(save_path + '_AFBM Model')
# Validation / Evaluation per Label

pn_model.load_weights('MLCPNBestWeights.h5')

for i in range(1,10):
    t = i / 10
    print(f"Theshold: {t}")
    data = []
    pn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE, epsilon=EPS),
        metrics=[
            PerLabelMetric(num_labels=NUM_CLASSES,threshold=t),
            ],
            run_eagerly=True,
        )
    data=pn_model.evaluate(x=val_ds)
    metrics = data[1]
    metrics = pd.DataFrame(metrics).T
    histfile = save_path + '_label_validation_allmets_autoweightsFixed_' + str(t) + '.csv'

    with open(histfile, mode='w') as f:
        metrics.to_csv(f)
