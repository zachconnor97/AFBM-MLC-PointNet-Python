# Plots3_12_autoweights\
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

metrics_names = ['Accuracy','Precision', 'Recall', 'F1 Score']
amarker = ['o', '^', 'X', '*']
dataframes = []
labels = [
    'ConvertCEtotoKE,AE,TE', 'ConvertEEtoAE', 'ConvertEEtoLE',
    'ConvertKEtoAE', 'ConvertKEtoEE', 'ConvertLEtoCE',
    'ConvertLEtoEE', 'ExportAE', 'ExportAEtoTE',
    'ExportCE', 'ExportEE', 'ExportGas', 'ExportLE',
    'ExportLiquid', 'ExportSolid', 'ImportEE',
    'ImportGas', 'ImportHE', 'ImportKE',
    'ImportLE', 'ImportLiquid', 'ImportSolid',
    'StoreGas', 'StoreLiquid', 'StoreSolid'
]
df = pd.DataFrame()
folder = str("C:/Users/Zachariah Connor/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/3_12_autoweights/")
path = str("MLCPN_Results_Autoweights.csv")
file = pd.read_csv(folder+path, header = None)    
file.reset_index(drop=True, inplace=True)

# Rename the columns for easier access
for i, metrics in enumerate(metrics_names):
    file = file.rename(columns={i+7: metrics })
# Also rename the rows
#file = file.rename(index=dict(zip(file.index, labels)))
file = file.drop(file.index[0])
df = file
#df = pd.concat((df, file), axis=0)
print(df)
 

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
#plt.savefig(f"C:/Users/Zachariah Connor/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/3_12_autoweights/Plots/{metric}_plot.svg", transparent=True, format='svg')
plt.show()
#plt.close()
#print("Plots saved successfully.")
