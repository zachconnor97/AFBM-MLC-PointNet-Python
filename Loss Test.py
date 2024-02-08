import numpy as np
import tensorflow as tf

yt = [[0,1,0,1,0],[1,1,1,1,0]]
yp = [[1.0,0.0,1.0,0.0,1.0],[0.0,0.0,0.0,0.0,0.0]]
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
loss = bce(yt,yp).numpy()
print('Loss=' + str(loss))