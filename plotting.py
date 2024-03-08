import matplotlib.pyplot as plt
import pandas as pd

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
metrics_names = ['accuracy', 'precision', 'recall', 'f1']
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
    file = pd.read_csv(str("C:/Users/gabri/OneDrive - Oregon State University/AFBM_TF_DATASET/MLCPN_Validation2024-03-07_16_5000_30_Learning Rate_0.001_Epsilon_1e-07_label_validation_allmets"+ str(threshold) + ".csv"), header = None)
    #file = pd.read_csv(str("C:/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/MLCPN_Validation2024-03-07_16_5000_30_Learning Rate_0.001_Epsilon_1e-07_label_validation_allmets" + str(threshold) + ".csv"))
    file.reset_index(drop=True, inplace=True)
    
    # Rename the columns for easier access
    for i, metrics in enumerate(metrics_names):
        file = file.rename(columns={i+5: metrics })
    # Also rename the rows
    file = file.rename(index=dict(zip(file.index, labels)))
    file = file.drop(file.index[0])
    df = pd.concat((df, file), axis=0)
#print(df)

label_dict_data = {}
for label in labels:
    label_dict_data[label] = df[df.index == label]
print(label_dict_data['ConvertCE']['accuracy'])

for flabel in labels:
    plt.figure()
    for metric in metrics_names:
        data = label_dict_data[label][metric].astype(float)
        #print(data)
        plt.plot(thresholds, data, label=metric)
    plt.legend()
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs. Threshold for' + str(flabel))
    plt.savefig(f"{flabel}+Learning Rate_0.001_plot.png")
plt.close()
#print("Plots saved successfully.")