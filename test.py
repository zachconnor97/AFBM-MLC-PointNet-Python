import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import date 
import os
import csv
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import pointnet, OrthogonalRegularizer, orthogonal_regularizer_from_config
from utils import PerLabelMetric, wbce_loss
from dataset import generate_dataset
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
save_path = str('/mnt/c/Users/' + username +'/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/MLCPN_Validation_New' + str(date.today()) + '_' + str(BATCH_SIZE) + '_' + str(NUM_POINTS) + '_' + str(NUM_EPOCHS) + '_Learning Rate_' + str(LEARN_RATE) + '_Epsilon_' + str(EPS))

metrics_names = ['Accuracy','Precision', 'Recall', 'F1 Score']
amarker = ['o', '^', 'X', '*']
dataframes = []
labels = [
    'ConvertCEtoKE,AE,TE', 'ConvertEEtoAE', 'ConvertEEtoLE',
    'ConvertKEtoAE', 'ConvertKEtoEE', 'ConvertLEtoCE',
    'ConvertLEtoEE', 'ExportAE', 'ExportAE,TE',
    'ExportCE', 'ExportEE', 'ExportGas', 'ExportLE',
    'ExportLiquid', 'ExportSolid', 'ImportEE',
    'ImportGas', 'ImportHE', 'ImportKE',
    'ImportLE', 'ImportLiquid', 'ImportSolid',
    'StoreGas', 'StoreLiquid', 'StoreSolid'
]

def validate(pn_model, val_ds, label_weights): # X is points and Y is labels
    stacked_loss = 0 
    for step, (xbt, ybt) in enumerate(val_ds):
        #print(f"Step: {step}")
        with tf.GradientTape() as t:
            # Trainable variables are automatically tracked by GradientTape
            current_loss = wbce_loss(ybt, pn_model(xbt))
            stacked_loss = stacked_loss + current_loss
        #print(f"Current Loss: {current_loss}")
    return stacked_loss/step


# Load mlc-PointNet model
pn_model = pointnet(num_points=NUM_POINTS, num_classes=NUM_CLASSES, train=False)

# Load dataset
train_ds, val_ds, label_weights = generate_dataset(filename=database)
print(f"Label Weights: {label_weights}")


"""
# Manually adjust weights of each label using the following code
# i = 1 # index of label
# w = 10 # weight value
# label_weights[i] = w 
# print(f"Adjusted Label Weights: {label_weights}")
"""

# Loads pretrained weights
pn_model.load_weights('MLCPNBestWeights.h5')

# Validate Model
t = 0.5
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
histfile = save_path + '_label_validation_' + str(t) + '.csv'
with open(histfile, mode='w') as f:
    metrics.to_csv(f)

file = metrics

# Rename the columns for easier access
for i, metrics in enumerate(metrics_names):
    file = file.rename(columns={i+7: metrics })
# Also rename the rows
#file = file.rename(index=dict(zip(file.index, labels)))
#file = file.drop(file.index[0])
df = file

# Add plotting for the bar charts (accuracy, precision, recall, F1)
label_dict_data = df

i = 0
width = 1
x = np.arange(len(labels))
for metric in metrics_names:
    plt.figure(figsize=(12, 6), dpi=100)
    #plt.borde(bottom=0.3) 
    data = label_dict_data[metric].astype(float)
    recs = plt.bar(labels, data)
    plt.bar_label(recs,padding=3,rotation=45)
    plt.legend()
    plt.ylim(top=1.1)
    plt.xlabel('Labels')
    plt.ylabel('Metric Value')
    plt.title(metric)
    plt.xticks(rotation=90)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    i += 1
plt.show()