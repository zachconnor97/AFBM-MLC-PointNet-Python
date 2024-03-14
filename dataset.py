import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import open3d
import pandas as pd
from tensorflow.keras.metrics import Metric


class DataSet:
    def __init__(self, num_points=2000, sample_ratio = 10000, batch_size=32, username='Zachariah' ) -> None:
        """_summary_

        Args:
            NUM_POINTS (int, optional): _description_. Defaults to 2000.
            SAMPLE_RATIO (int, optional): _description_. Defaults to 10000.
            BATCH_SIZE (int, optional): _description_. Defaults to 32.
            username (str, optional): _description_. Defaults to 'Zachariah'.
        """
        self.num_points_ = num_points
        self.sample_ratio_ = sample_ratio//self.num_points_
        self.batch_size_ = batch_size
        self.username_ = username

    def pc_read(self, path):
        """_summary_
        This funtion will read the point cloud of the desired object at the specified path and adjust the number of points to self.num_points_input
        and self.sample_ratio_.

        Args:
            path (string): Address of the object to read its point cloud

        Returns:
            cloud (numpy array): point cloud of the 
        """
        
        # cloud_path_header = str('/mnt/c/' + self.username + '/Oregon State University/Connor, Zachariah Ayhn - AllClouds10k/')
        # Use second one for WSL
        cloud_path_header = str('/mnt/c/Data/PointNetWork/AllClouds10k/')
        try:
            path = path.numpy()
            path = np.array2string(path)
            for character in "[]]'":
                path = path.replace(character, '')
            path = path[1:]
            path = cloud_path_header + path 
            cloud = open3d.io.read_point_cloud(path)
            cloud = cloud.uniform_down_sample(every_k_points=int(self.sample_ratio_ ))
        except:
            cloud = np.random.rand((self.num_points_,3))
            path = 'ERROR IN PCREAD: Transformation from Tensor to String Failed'
        finally:
            cloud = cloud.points
            cloud = np.asarray([cloud])[0]
        if len(cloud) <= 0:
            print(path)
        return cloud

    # ISparse Matrix Encoding Function
    def Sparse_Matrix_Encoding(self, df):
        """_summary_

        Args:
            df (Pandas Dataframe):              Input dataframe containing all the labels

        Returns:
            sparse_matrix (Pandas Dataframe):   Output dataframe with encoded labels for the classes
                                                (Encoded for multi label classification)
        """
    
        # Get list/array/whatever of unique labels
        uniquelabels = df.stack().unique()
        uniquelabels.sort()
        uniquelabels = np.delete(uniquelabels,len(uniquelabels)-1,0)
        
        # Encode all of the labels to the point cloud index as a length(dataframe) by length(uniquelabels) sparse matrix (1 or 0 only)
        encodedLabel = np.zeros((len(df), len(uniquelabels)), dtype=float)
        # Loop through clouds and labels
        for i in range(len(df)):
            for j, label in enumerate(df.columns):
                req_index = np.where(uniquelabels == df.iloc[i, j])[0]
                if req_index.size > 0:
                    req_index = req_index[0]
                    if df.iloc[i, j] == "nan":
                        encodedLabel[i, req_index] = 0
                    else:
                        encodedLabel[i, req_index] = 1
        sparse_matrix = encodedLabel
        return sparse_matrix

    def augment(self, points):
        """_summary_

        Args:
            points (numpy array): Point Cloud data of an object

        Returns:
            points: Augmented point cloud data
        """
        points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
        # shuffle points
        points = tf.random.shuffle(points)
        return points

    def generate_dataset(self, filename):
        """_summary_

        Args:
            filename (string):          Address for full dataset file containing the location and labels

        Returns:
            train_ds (dataset):         Training Dataset
            val_ds (dataset):           Validation Dataset
            label_weights(dictionary):  Weights for each label for the weighted loss function

        """
        # Import the csv and convert to strings
        df = pd.read_csv(filename)
        df = df.astype('str')
        
        # Seperates cloud paths to pandas series 
        file_paths = df.pop('cloudpath')
        #print(file_paths)
        # Removes non-necessary dataframe columns to leave just FBM labels
        df.pop('SysetID')
        df.pop('Name')
        df.pop('.obj paths')
        df.pop('fileid')
        df.pop('status')
        num_files = float(len(df))
        sparse_matrix = self.Sparse_Matrix_Encoding(df) 
        df = []
        label_counts = sparse_matrix.sum(axis=0)
        label_weights = (num_files / (25 * label_counts))
        label_weights = {k: v for k, v in enumerate(label_weights)}
        #print(type(label_weights))
        #print(label_weights)

        # Slice file paths and labels to tf.data.Dataset
        file_paths = np.asmatrix(file_paths)
        nfile_paths = file_paths.reshape((np.size(file_paths),1)) 
        nfile_paths = np.asarray(nfile_paths)
        tfile_paths = tf.constant(nfile_paths.tolist())
        tsparse = tf.constant(sparse_matrix.tolist())
        fileset = tf.data.Dataset.from_tensor_slices((tfile_paths))
        labelset = tf.data.Dataset.from_tensor_slices((tsparse))
        
        train_points = fileset.skip(int(0.3*len(fileset)))
        train_label = labelset.skip(int(0.3*len(labelset)))
        
        val_points = fileset.take(int(0.3*len(fileset)))
        val_label = labelset.take(int(0.3*len(labelset)))
        
        val_points = val_points.map(lambda x: tf.py_function(self.pc_read, [x], tf.float64))
        train_points = train_points.map(lambda x: tf.py_function(self.pc_read, [x], tf.float64))
        train_points = train_points.map(lambda x: tf.py_function(self.augment, [x], tf.float64))

        #val_ds = tf.data.Dataset.zip((val_points, val_label))
        #train_ds = tf.data.Dataset.zip((train_points, train_label))
        val_ds = tf.data.Dataset.zip((val_points, val_label))
        train_ds = tf.data.Dataset.zip((train_points, train_label))
        val_ds = val_ds.batch(self.batch_size_)
        train_ds = train_ds.batch(self.batch_size_) # ADDS A lot of time .shuffle(buffer_size=20000,reshuffle_each_iteration=True)

        #Testing stuff
        """
        data = afbm_dataset.take(1)
        points, labels = list(data)[0]
        #print(labels)
        print(points.numpy())
        print(type(points.numpy()))
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points.numpy())
        open3d.visualization.draw_geometries([pcd])
        """
        return train_ds, val_ds, label_weights