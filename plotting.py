import matplotlib.pyplot as plt
import pandas as pd

# accuracy is column 6
    # precision is colum 7
    # recall is column 8
    # F1 is column 9

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
metrics = ['accuracy', 'precision', 'recall', 'f1']
dataframes = []
labels = [
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
for threshold in thresholds:
    file = pd.read_csv(str("C:/Users/gabri/OneDrive - Oregon State University/AFBM_TF_DATASET/2024-02-23_16_5000_25_Learning Rate_2.5e-05_Epsilon 1e-07_label_validation_allmets_2_pretty pretty good.csv"))
    #file = pd.read_csv(str("C:/Users/Zachariah/OneDrive - Oregon State University/Research/AFBM/AFBM Code/AFBMGit/AFBM_TF_DATASET/MLCPN_Validation2024-03-07_16_5000_30_Learning Rate_0.001_Epsilon_1e-07_label_validation_allmets" + str(threshold) + ".csv"))
    df = pd.concat((df, file), axis=0)
    
print(df)
"""
for metric in metrics:
    plt.figure(figsize=(8, 6))
    for i in range(0,24):
        plt.plot(thresholds, df[metric], label=labels[i])
    plt.title(f"{metric.capitalize()} vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()
    plt.savefig(f"path_to_save_plots/{metric}_vs_threshold.png")  # Save the plot
    plt.close()  # Close the current figure to release memory


print("Plots saved successfully.")

"""