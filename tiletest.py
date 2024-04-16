import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

a = tf.constant([[1,2,3],[4,5,6]], tf.int32)
b = tf.constant([2,1], tf.int32)
c = tf.tile(a, 2)
print(c)