import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

tf.random.set_seed(1234)

clouds = tf.random.normal([32,5000,3])

inputs = keras.Input(shape=(5000, 3))
x = layers.Dense(64)(inputs)
x = layers.BatchNormalization(momentum=0.0)(x)
x = layers.Activation("relu")(x)
#print(x)
x = layers.Dense(128)(x)
x = layers.BatchNormalization(momentum=0.0)(x)
x = layers.Activation("relu")(x)

x = layers.Dense(1024)(x)
x = layers.BatchNormalization(momentum=0.0)(x)
x = layers.Activation("relu")(x)

x = layers.GlobalMaxPooling1D()(x)

x = layers.Dense(512)(x)
x = layers.BatchNormalization(momentum=0.0)(x)
x = layers.Activation("relu")(x)

x = layers.Dense(256)(x)
x = layers.BatchNormalization(momentum=0.0)(x)
x = layers.Activation("relu")(x)

x = layers.Dense(
        3 * 3,
        kernel_initializer="zeros",
        bias_initializer = keras.initializers.Constant(np.eye(3).flatten()),
        #activity_regularizer = OrthogonalRegularizer(3),
    )(x)

x = layers.Reshape((3, 3))(x)

x = layers.Dot(axes=(2, 1))([inputs, x])
print(x)
outputs = layers.Dense(25, activation="softmax")(x)
model = keras.Model(inputs=inputs,outputs=x)
model.summary()
model.predict(clouds)

#print(y)