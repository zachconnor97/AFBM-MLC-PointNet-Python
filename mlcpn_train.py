import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import date 
from model import mlcpn, Generator
from utils import PerLabelMetric, GarbageMan
from dataset import DataSet
import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class mlcpn_train:
    def __init__(self, num_points = 2000, classes=25, training=True, lr = 0.0003, batch_size=32, epochs=25, username='Zachariah', database= "AFBMData_NoChairs_Augmented.csv", sav_path = 'C:/Data/PointNetWork/Output_files/') -> None:
        """_summary_

        Args:
            num_points (int, optional):     Number of points in a point cloud. Defaults to 2000.
            classes (int, optional):        Number of labels in the dataset. Defaults to 25.
            training (bool, optional):      Whether to train the network. Defaults to True.
            lr (float, optional):           Learning Rate. Defaults to 0.0003.
            batch_size (int, optional):     Batch size. Defaults to 32.
            epochs (int, optional):         Number of epochs. Defaults to 25.
            username (str, optional):       Username of the host. Defaults to 'Zachariah'.
            database (str, optional):       CSV database of labelled data used to train the network. Defaults to "AFBMData_NoChairs_Augmented.csv".
            sav_path (str, optional):       Address to store the output files. Defaults to 'C:/Data/PointNetWork/Output_files/'.
        """
        self.num_points_ = num_points
        self.num_classes_ = classes
        self.training_ = training
        self.learn_rate_ = lr
        self.batch_size_ = batch_size
        self.num_epochs_ = epochs
        self.username_ = username
        self.label_database_ =  database
        self.output_path_ = str(sav_path + str(date.today()) + '_' + str(self.batch_size_) + '_' + str(self.num_points_) + '_' + str(self.num_epochs_) + '_' + 'Learning Rate_' + str(self.learn_rate_))
        
    def training(self):
        """_summary_
        This function is used to import and divide the Point clouds using the DataSet class. It is divided into training(train_ds) and validation(val_ds) sets.
        """
        dataset_in = DataSet(self.num_points_, 10000, self.batch_size_, self.username_)
        train_ds, val_ds, label_weights = dataset_in.generate_dataset(filename=self.database_)
        with open("Label_Weights.csv", mode='w') as f:
            writer = csv.writer(f)
            for key, value in label_weights.items():
                writer.writerow([key, value])
        pt_net = mlcpn(self.num_classes_, self.num_points_, self.training_)
        pn_model = pt_net.pointnet()
        pn_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=self.learn_rate_),
        metrics=[
            #PerLabelMetric(num_labels=NUM_CLASSES),
            tf.keras.metrics.BinaryAccuracy(threshold=0.5),
            tf.keras.metrics.Precision(thresholds=[0.5,1]),
            tf.keras.metrics.Recall(thresholds=[0.5,1]),
            ],      
        run_eagerly=True,
        )

        pn_model.fit(x=train_ds, epochs=self.num_epochs_, class_weight=label_weights, validation_data=val_ds, callbacks=[GarbageMan()])
        pn_model.save(self.save_path_ + '_AFBM Model')

        ## Save history file
        histdf = pd.DataFrame(pn_model.history.history)
        histfile = self.save_path_ + '_train_history_per_label_met.csv'
        with open(histfile, mode='w') as f:
            histdf.to_csv(f)

        histdf = pd.DataFrame(pn_model.history.history)
        histfile = self.save_path_ + '_train_history.csv'
        with open(histfile, mode='w') as f:
            histdf.to_csv(f)

        # Validation / Evaluation per Label
        data = []
        pn_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate= self.learn_rate_),
            metrics=[
                PerLabelMetric(num_labels=self.num_classes_),
                ],
                run_eagerly=True,
            )
        data=pn_model.evaluate(x=val_ds)
        metrics = data[1]
        metrics = pd.DataFrame(metrics).T
        print(metrics)
        histfile = self.save_path_ + '_label_validation_allmets_2.csv'

        with open(histfile, mode='w') as f:
            metrics.to_csv(f)


if __name__ == '__main__':
    model_train = mlcpn_train(500, 25, True, 0.003, 32, 2, 'shita', database= "/mnt/c/Data/PointNetWork/AFBMData_NoChairs_Augmented.csv", sav_path='/mnt/c/Data/PointNetWork/Output_files/')
    model_train.training()
    