# AFBM MLC-PointNet Python
Automated Functional Basis Modeling using MLC-PointNet. Code for work by Zachariah Connor, Shital Sable, Gabriel Connor, Dr. Matthew Campbell, and Dr. Robert Stone. Publications forthcoming. Please cite our publications if you use this code. 

Tested on WSL Ubuntu 2204.2.33.0, Python 3.10.12, TensorFlow 2.15.0, Keras 2.15.0. Requires ~12GB of VRAM, we tested on Nvidia RTX 3060. Training took approximately 1.5hr on a computer running WSL 2 in Windows 11 Pro with Nvidia RTX 3060, Intel i5 12600k, and 32GB DDR4 RAM.

-AFBMData_NoChairs_Augmented.csv | CSV file with the AFBM Dataset file names and Functional Basis Labels

-MLCPNBestWeights.h5 | Weights for MLC-PointNet that were trained on the AFBM Dataset

-dataset.py | Utility file that reads the point clouds 

-gcam_mlcpn.py | Uses a modified GradCAM algorithm to predict the locations of functions on the point cloud 

-model.py | MLC-PointNet model is defined here

-train.py | Trains MLC-PointNet and saves the weights of the network

-test.py | Validate trained MLC-PointNet network

-utils.py | Metric and Loss Functions

-predict.py | Use MLC-PointNet to predict the Functional Basis labels on a point cloud


