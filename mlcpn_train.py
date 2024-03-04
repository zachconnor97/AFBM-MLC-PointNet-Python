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
    def __init__(self, num_points = 2000, classes=25, training=True, lr = 0.0003, batch_size=32, epochs=25, username='Zachariah', database=  "AFBMData_NoChairs_Augmented.csv", sav_path = 'C:/Data/PointNetWork/Output_files/') -> None:
        self.num_points_ = num_points
        self.num_classes_ = classes
        self.training_ = True
        self.learn_rate_ = lr
        self.batch_size_ = batch_size
        self.num_epochs_ = epochs
        self.username_ = 'Zachariah'
        self.database_ =  database
        self.save_path_ = str(sav_path + str(date.today()) + '_' + str(self.batch_size_) + '_' + str(self.num_points_) + '_' + str(self.num_epochs_) + '_' + 'Learning Rate_' + str(self.learn_rate_))
        
    def training(self) -> int:
        ds1 = DataSet()
        train_ds, val_ds, label_weights = ds1.generate_dataset(filename=self.database_)
        with open("Label_Weights.csv", mode='w') as f:
            writer = csv.writer(f)
            for key, value in label_weights.items():
                writer.writerow([key, value])
        pn = mlcpn(self.num_classes_, self.num_points_, self.training_)
        pn_model = pn.pointnet()
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
        histdf = pd.DataFrame(pn_model.history[0]).T
        histfile = self.save_path_ + '_train_history_per_label_met.csv'
        with open(histfile, mode='w') as f:
            histdf.to_csv(f)

        histdf = pd.DataFrame(pn_model.history[1:])
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
    model_train = mlcpn_train(2000, 25, True, 0.0003, 32, 25, 'Zachariah', database=  "/mnt/c/Data/PointNetWork/AFBMData_NoChairs_Augmented.csv", sav_path='/mnt/c/Data/PointNetWork/Output_files/')
    model_train.training()
    