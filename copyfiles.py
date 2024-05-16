import shutil
import pandas as pd
import os
import csv

filename = "C:/Users/gabri/OneDrive - Oregon State University/Z Research/test.csv"

df = pd.read_csv(filename)
print(df)

# Destination directory
destination_dir = 'C:/Users/gabri/OneDrive - Oregon State University/Z Research/CopyTests'


os.makedirs(destination_dir, exist_ok=True)


with open(filename, 'r') as file:
    reader = csv.reader(file)
    for path in reader:
        path = path[0]  
        if os.path.isfile(path):
            file_name = os.path.basename(path)  
            destination_file = os.path.join(destination_dir, file_name)
            
            # Copy file from source to destination
            shutil.copy(path, destination_file)
            print(f"File {file_name} copied successfully.")
        else:
            print(f"File {path} does not exist.")