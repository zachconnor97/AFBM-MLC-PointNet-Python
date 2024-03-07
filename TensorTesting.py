import tensorflow as tf
import numpy as np
#tf.config.run_functions_eagerly(False)

x = tf.constant([5, 4, 6])
y = tf.constant([5, 2, 5])
tab = tf.math.greater(x, y)

print(tab)
