# Plots
import matplotlib.pyplot as plt
import pandas as pd


metrics_names = ['precision', 'recall', 'f1']
amarker = ['o', '^', 'X']
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
#for threshold in thresholds:
    #file = pd.read_csv(str("C:/Users/gabri/OneDrive - Oregon State University/AFBM_TF_DATASET/MLCPN_Validation2024-03-07_16_5000_30_Learning Rate_0.00025_Epsilon_ 1e-07_label_validation_allmets_"+ str(threshold) + ".csv"), header = None)
    #file = pd.read_csv(str("C:/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/MLCPN_Validation2024-03-07_16_5000_30_Learning Rate_0.001_Epsilon_1e-07_label_validation_allmets" + str(threshold) + ".csv"))
folder = str("C:/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/3_12_autoweights/")
path = str("MLCPN_ResultsAutoweights.csv")
file = pd.read_csv(folder+path, header = None)    
file.reset_index(drop=True, inplace=True)

# Rename the columns for easier access
for i, metrics in enumerate(metrics_names):
    file = file.rename(columns={i+8: metrics })
# Also rename the rows
file = file.rename(index=dict(zip(file.index, labels)))
file = file.drop(file.index[0])
df = pd.concat(df)
#df = pd.concat((df, file), axis=0)
print(df)
 

label_dict_data = df
 
plt.figure(figsize=(8, 6), dpi=100)
for metric in metrics_names:
    data = label_dict_data[metric].astype(float)
    #print(f"For {flabel}, {metric}: Length of thresholds={len(thresholds)}, Length of data={len(data)}")
    #print(data)
    #print(thresholds)
    plt.plot(data, label=metric, linewidth = 1.0, marker=amarker[i], markersize = 7.0, linestyle='dotted', color='k')
plt.legend()
plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.ylim(top=1.0)
plt.title('')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)    
#plt.savefig(f"C:/Users/Zachariah Connor/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/3_12_autoweights/Plots/{metric}_plot.svg", transparent=True, format='svg')
plt.show()
#plt.close()
#print("Plots saved successfully.")