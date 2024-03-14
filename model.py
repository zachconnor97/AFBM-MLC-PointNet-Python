import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

class mlcpn:
    def __init__(self, classes, num_points=2000, training= True) -> None:
        """_summary_

        Args:
            classes ():                 Classes for classification
            num_points (int):           Number of point in the point cloud. Defaults to 2000.
            training (bool, optional):  Whether to train the network or not. Defaults to True.
        """
        self.num_classes = classes
        self.num_points = num_points
        self.training = training


    def conv_bn(self, x, filters, stride=1, train=True):
        """_summary_

        Args:
            x (_type_):             Input Data
            filters (_type_):       Size of filter
            stride (int, optional): Stride for the convolution layer. Defaults to 1.
            train (bool, optional): Whether to train the network or not. Defaults to True.

        Returns:
            output layer
        """
        x = layers.Conv1D(filters, strides=stride, kernel_size=1, padding="valid")(x)
        x = layers.BatchNormalization(momentum=0.0, trainable=train)(x)
        return layers.Activation('LeakyReLU')(x)


    def dense_bn(self, x, filters, train=True):
        """_summary_

        Args:
            x (_type_):             Input Data
            filters (_type_):       Size of filter
            train (bool, optional): Whether to train the layer or not. Defaults to True.

        Returns:
            output layer
        """
        x = layers.Dense(filters)(x)
        x = layers.BatchNormalization(momentum=0.0, trainable=train)(x)
        return layers.Activation('LeakyReLU')(x)


    def orthogonal_regularizer_from_config(self, config):
        return OrthogonalRegularizer(**config)


    def tnet(self, inputs, num_features, train=True):
        """_summary_

        Args:
            inputs (): Input Data
            num_features (): Number of features
            train (bool, optional): Whether to train the layer or not. Defaults to True.

        Returns:
            output layer
        """
        bias = keras.initializers.Constant(np.eye(num_features).flatten())
        reg = OrthogonalRegularizer(num_features)
        x = self.conv_bn(inputs, 64, train=train)
        x = self.conv_bn(x, 128, train=train)
        x = self.conv_bn(x, 1024, train=train)
        x = layers.GlobalMaxPooling1D()(x)
        x = self.dense_bn(x, 512, train=train)
        x = self.dense_bn(x, 256, train=train)
        x = layers.Dense(
            num_features * num_features,
            kernel_initializer="zeros",
            bias_initializer=bias,
            activity_regularizer=reg)(x)
        feat_T = layers.Reshape((num_features, num_features))(x)
        return layers.Dot(axes=(2, 1))([inputs, feat_T])

    def pointnet(self):
        """
        Returns Keras MLC-PointNet Implementation
        Args:
            training (boolean): Training or not
            num_points (scalar): Number of points in the point cloud
        """    
        # print(self.num_points)
        inputs = keras.Input(shape=(self.num_points, 3))
        x = self.tnet(inputs, 3, train=self.training)
        x = self.conv_bn(x, 64, train=self.training)
        x = self.conv_bn(x, 64, train=self.training)
        x = self.tnet(x, 64, train=self.training)
        x = self.conv_bn(x, 64, train=self.training)
        x = self.conv_bn(x, 128, train=self.training)
        x = self.conv_bn(x, 1024, train=self.training)
        x = layers.GlobalMaxPooling1D()(x)
        x = self.dense_bn(x, 512, train=self.training)
        x = self.dense_bn(x, 256, train=self.training)
        outputs = layers.Dense(self.num_classes, activation="sigmoid")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
        return model

class Generator:
    def __init__(self, num_points, num_classes, train) -> None:
        """_summary_:

        Args:
            num_points (_type_):    Number of points in point cloud
            num_classes (_type_):   Number of classes in the dataset
            train (_type_):         Whether to train the layer or not. Defaults to True.
        """
        self.num_classes = num_classes
        self.num_points = num_points
        self.training = train

        
    def conv_bn(self, x, filters, stride=1, train=True):
        """_summary_

        Args:
            x (_type_):                 Input Data
            filters (_type_):           Filter size
            stride (int, optional):     Size of stride. Defaults to 1.
            train (bool, optional):     Whether to train the layer or not. Defaults to True.

        Returns:
            Output Layer
        """
        x = layers.Conv1D(filters, strides=stride, kernel_size=1, padding="valid")(x)
        x = layers.BatchNormalization(momentum=0.0, trainable=train)(x)
        return layers.Activation('LeakyReLU')(x)

    def dconv_bn(self, x, filters, stride=1, train=True):
        """_summary_

        Args:
            x (_type_):                 Input Data
            filters (_type_):           Filter size
            stride (int, optional):     Size of stride. Defaults to 1.
            train (bool, optional):     Whether to train the layer or not. Defaults to True.

        Returns:
            Output Layer
        """
        x = layers.Conv1DTranspose(filters, kernel_size=1, strides=stride, padding="valid")(x)
        x = layers.BatchNormalization(momentum=0.0, trainable=train)(x)
        return layers.Activation('LeakyReLU')(x)

    def dense_bn(self, x, filters, train=True):
        """_summary_

        Args:
            x (_type_):                 Input Data
            filters (_type_):           Filter size
            train (bool, optional):     Whether to train the layer or not. Defaults to True.

        Returns:
            Output Layer
        """
        x = layers.Dense(filters)(x)
        x = layers.BatchNormalization(momentum=0.0, trainable=train)(x)
        return layers.Activation('LeakyReLU')(x)

    def generator(self):
        """
        Returns Keras C-GAN
        Args:
            training (boolean): Training or not
            num_points (integer): Number of points in the point cloud
            num_classes (integer): Number of classes         
        """    
        input = keras.Input(shape=(self.num_classes,1))
        x = self.conv_bn(input, 1028, train=self.training)
        x = self.conv_bn(x, 512, train=self.training)
        x = self.conv_bn(x, 256, train=self.training)
        x = self.conv_bn(x, 128, train=self.training)
        x = self.conv_bn(x, 1, stride=self.num_classes, train=self.training)
        x = self.dconv_bn(x, 1, stride= self.num_points, train=self.training)
        x = self.dconv_bn(x,  self.num_points/8, train=self.training)
        x = self.dconv_bn(x,  self.num_points/4, train=self.training)
        x = self.dconv_bn(x,  self.num_points/2, train=self.training)
        x = self.dconv_bn(x,  self.num_points, train=self.training)
        #x = layers.Reshape((5000,3))(x)
        x = layers.Dense(3, activation="sigmoid")(x)
        model = keras.Model(inputs=input, outputs=x, name="c_gan")
        return model

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features=None, l2reg=0.001, **kwargs):
        super(OrthogonalRegularizer, self).__init__(**kwargs)
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
        return{'num_features': self.num_features,'l2reg': self.l2reg}

    @classmethod
    def from_config(cls, config):
        return cls(num_features=config.pop('num_features', None), **config)
    
