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
pn_model = pointnet(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=True)
#print(pn_model.get_weights()[0])
#print(pn_model.get_weights()[1])
EStop = EarlyStopping(monitor='val_loss',patience=3, mode='min')


def loss(target_y, predicted_y, label_weights=None):
    # Update to binary cross entropy loss
    target_y = tf.cast(target_y, dtype=tf.float32)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss = bce(target_y, predicted_y).numpy() 
    # print(loss)
    # print(type(loss))
    return loss

def wbce_loss(target_y, predicted_y, label_weights=None):
    from keras.src import backend, backend_config
    epsilon = backend_config.epsilon
    lw=np.array(list(label_weights.items()))
    lw = lw[:,1]
    target = tf.convert_to_tensor(target_y, dtype='float32')
    output = tf.convert_to_tensor(predicted_y, dtype='float32')
    epsilon_ = tf.constant(epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    bceloss = target * tf.math.log(output + epsilon())
    bceloss += (1-target) * tf.math.log(1 - output + epsilon())
    wbceloss = backend.mean(-bceloss * lw) 
    print(wbceloss.numpy())
    print(type(wbceloss.numpy()))
    return wbceloss.numpy()

def train(pn_model, train_ds, learn_rate, label_weights=None): # X is points and Y is labels
    stacked_loss = 0 
    for step, (xbt, ybt) in enumerate(train_ds):
        print(f"Step: {step}")
        with tf.GradientTape() as t:
            # Trainable variables are automatically tracked by GradientTape
            current_loss = loss(ybt, pn_model(xbt))
            #current_loss = wbce_loss(ybt, pn_model(xbt), label_weights)
            stacked_loss = stacked_loss + current_loss
        print(f"Current Loss: {current_loss}")
        grads = t.gradient(current_loss, pn_model.trainable_weights)    
        # Subtract the gradient scaled by the learning rate
        g_optimizer.apply_gradients(zip(grads, pn_model.trainable_weights))
        #g_optimizer.apply_gradients(zip(grads*learn_rate, pn_model.trainable_weights))
    return stacked_loss/step

def validate(pn_model, val_ds, label_weights): # X is points and Y is labels
    stacked_loss = 0 
    for step, (xbt, ybt) in enumerate(val_ds):
        print(f"Step: {step}")
        with tf.GradientTape() as t:
            # Trainable variables are automatically tracked by GradientTape
            current_loss = wbce_loss(ybt, pn_model(xbt), label_weights)
            stacked_loss = stacked_loss + current_loss
        print(f"Current Loss: {current_loss}")
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
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch}:")
        # Update the model with the single giant batch
        e_loss = train(pn_model, train_ds, label_weights=label_weights, learn_rate=LEARN_RATE)
        # Track this before I update
        weights.append(pn_model.get_weights()[0])
        biases.append(pn_model.get_weights()[1])
        print(f"W = {pn_model.get_weights()[0]}, B = = {pn_model.get_weights()[1]}")
    print(f"Validation Loss: {validate(pn_model, val_ds, label_weights)}")
        


#Callback for saving best model
model_checkpoint = ModelCheckpoint(
    filepath=save_path,
    monitor='val_loss',  # Monitor validation loss
    save_best_only=True,  # Save only the best model
    mode='min',  # Save when validation loss is minimized
    verbose=1  # Show information about saving
)

train_ds, val_ds, label_weights = generate_dataset(filename=database)

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

# pn_model Code for the training loop
weights = []
biases = []


#current_loss = loss(y = train_points, pn_model(train_label))

print(f"Starting:")
#print("    ", report(pn_model, current_loss=1))
training_loop(pn_model, train_ds, val_ds, label_weights)

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