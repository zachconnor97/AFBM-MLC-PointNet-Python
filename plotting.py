import matplotlib as plt
import pandas as pd

# accuracy is column 6
    # precision is colum 7
    # recall is column 8
    # F1 is column 9

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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


for threshold in thresholds:
    df = pd.read_csv(f"C:/Users/gabri/OneDrive - Oregon State University/AFBM_TF_DATASET/2024-02-23_16_5000_25_Learning Rate_2.5e-05_Epsilon 1e-07_train_history_per_label_met.csv", header = None)
    dataframes.append(thresholds)
    dataframes.append(df)

print(dataframes)

for metric in metrics:
    plt.figure(figsize=(8, 6))
    for i in range(0,24):
        plt.plot(df['threshold'], df[metric], label=labels[i])
    plt.title(f"{metric.capitalize()} vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()
    plt.savefig(f"path_to_save_plots/{metric}_vs_threshold.png")  # Save the plot
    plt.close()  # Close the current figure to release memory


print("Plots saved successfully.")

