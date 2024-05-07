# AFBM MLC-PointNet Python
Automated Functional Basis Modeling using MLC-PointNet. Code for work by Zachariah Connor, Shital Sable, Gabriel Connor, Dr. Matthew Campbell, and Dr. Robert Stone. Publications forthcoming. Please cite our publications if you use this code. 

Tested on WSL Ubuntu 2204.2.33.0 and Python 3.10.12. 

-AFBMData_NoChairs_Augmented.csv | CSV file with the AFBM Dataset file names and Functional Basis Labels

-MLCPNBestWeights.h5 | Weights for MLC-PointNet that were trained on the AFBM Dataset

-dataset.py | Utility file that reads the point clouds 

-gcam_mlcpn.py | Uses a modified GradCAM algorithm to predict the locations of functions on the point cloud 

-model.py | MLC-PointNet model is defined here

-train.py | Trains MLC-PointNet and saves the weights of the network

-test.py | Validate trained MLC-PointNet network

-utils.py | Metric and Loss Functions

-predict.py | Use MLC-PointNet to predict the Functional Basis labels on a point cloud


