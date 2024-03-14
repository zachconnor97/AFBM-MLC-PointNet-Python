# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:55:21 2024

@author: gabri
"""

import matplotlib.pyplot as plt
import pandas as pd

folder = "C:/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/3_12_man_weights_1000/"
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
amarker = ['o', '^', 'X']
metrics_names = ['precision', 'recall', 'f1']
dataframes = []
labels = [
    'NA',
    'ConvertCE', 'ConvertEEtoAE', 'ConvertEEtoLE',
    'ConvertKEtoAE', 'ConvertKEtoEE', 'ConvertLEtoCE',
    'ConvertLEtoEE', 'ExportAE', 'ExportAEtoTE',
    'ExportCE', 'ExportEE', 'ExportGas', 'ExportLE',
    'ExportLiquid', 'ExportSolid', 'ImportEE',
    'ImportGas', 'ImportHE', 'ImportKE',
    'ImportLE', 'ImportLiquid', 'ImportSolid',
    'StoreGas', 'StoreLiquid', 'StoreSolid'
]
df = pd.DataFrame()
# The for loop successfully combines all of the csv files into a single dataframe. Starts with 0.1 and goes to 0.9
for threshold in thresholds:
    #file = pd.read_csv(str("C:/Users/gabri/OneDrive - Oregon State University/AFBM_TF_DATASET/MLCPN_Validation2024-03-07_16_5000_30_Learning Rate_0.00025_Epsilon_ 1e-07_label_validation_allmets_"+ str(threshold) + ".csv"), header = None)
    #file = pd.read_csv(str("C:/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/MLCPN_Validation2024-03-07_16_5000_30_Learning Rate_0.001_Epsilon_1e-07_label_validation_allmets" + str(threshold) + ".csv"))
    path = str("MLCPN_Validation_MOD_WEIGHTS2024-03-12_16_5000_30_Learning Rate_0.00025_Epsilon_1e-07_label_validation_allmets_" + str(threshold) + ".csv")
    file = pd.read_csv(folder+path, header=None)
    file.reset_index(drop=True, inplace=True)
    
    # Rename the columns for easier access
    for i, metrics in enumerate(metrics_names):
        file = file.rename(columns={i+6: metrics })
    # Also rename the rows
    file = file.rename(index=dict(zip(file.index, labels)))
    file = file.drop(file.index[0])
    df = pd.concat((df, file), axis=0)
#print(df)

label_dict_data = {}
for label in labels:
    label_dict_data[label] = df[df.index == label]


for flabel in labels:
    if flabel == 'NA':
        continue
    plt.figure(figsize=(8, 6), dpi=90)
    i = 0
    for metric in metrics_names:
        data = label_dict_data[flabel][metric].astype(float)
        #print(f"For {flabel}, {metric}: Length of thresholds={len(thresholds)}, Length of data={len(data)}")
        #print(data)
        #print(thresholds)
        plt.plot(thresholds, data, label=metric, linewidth = 1.0, marker=amarker[i], markersize = 7.0, linestyle='dotted', color='k')
        i += 1
    plt.legend()
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.ylim(bottom=0.0, top=1.0)
    plt.title('Metrics vs. Threshold for ' + str(flabel))
    plt.savefig((folder + flabel + "_plot.svg"))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
plt.close()
print("Plots saved successfully.")