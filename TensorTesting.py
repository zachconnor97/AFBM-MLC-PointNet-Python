import tensorflow as tf
import numpy as np
#tf.config.run_functions_eagerly(False)
tensor_array = tf.TensorArray(dtype=tf.Tensor, size=0,dynamic_size=True)
stringy = ['string1', 'string2','string3','string4','string5']
t = tf.constant(stringy,dtype=tf.string)
tensor_array = tensor_array.write(0,t)
print(tensor_array)
print(t)
print(type(t))
#t = tf.strings.to_number(t)
#ts = t.numpy()

test = tf.strings.as_string([3,2])
print(test.numpy())