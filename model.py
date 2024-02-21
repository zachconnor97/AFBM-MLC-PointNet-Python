import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

def conv_bn(x, filters, training=True):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0, trainable=training)(x)
    return layers.Activation('LeakyReLU')(x)

def dconv_bn(x, filters, training=True):
    x = layers.Conv1DTranspose(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0, trainable=training)(x)
    return layers.Activation('LeakyReLU')(x)

def dense_bn(x, filters, training=True):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0, training=training)(x)
    return layers.Activation('LeakyReLU')(x)

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
    
## Custom function to instantiate the regularizer during loading
def orthogonal_regularizer_from_config(config):
    return OrthogonalRegularizer(**config)

def tnet(inputs, num_features, training=True):
    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 64, training=training)
    x = conv_bn(x, 128, training=training)
    x = conv_bn(x, 1024, training=training)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 512, training=training)
    x = dense_bn(x, 256, training=training)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

def pointnet(num_points, num_classes, training=True):
    """
    Returns Keras MLC-PointNet Implementation
    Args:
        training (boolean): Training or not
        num_points (scalar): Number of points in the point cloud
    """    

    inputs = keras.Input(shape=(num_points, 3))
    x = tnet(inputs, 3, training=training)
    x = conv_bn(x, 64, training=training)
    x = conv_bn(x, 64, training=training)
    x = tnet(x, 64, training=training)
    x = conv_bn(x, 64, training=training)
    x = conv_bn(x, 128, training=training)
    x = conv_bn(x, 1024, training=training)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 512, training=training)
    x = dense_bn(x, 256, training=training)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")

def generator(num_points, num_classes, training=True):
    input = keras.Input(shape=(num_classes,1))
    x = dconv_bn(input, 512, training=training)
    x = dconv_bn(input, 256, training=training)
    x = dconv_bn(input, 128, training=training)
    x = dconv_bn(input, 64, training=training)
    x = dconv_bn(input, 1, training=training)
    model = keras.Model(inputs=input, outputs=x, name="c_gan")