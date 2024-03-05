import tensorflow as tf
from keras.src import backend, backend_config
epsilon = backend_config.epsilon

def wbce_loss(target_y, predicted_y, label_weights=None):
    from keras.src import backend, backend_config
    epsilon = backend_config.epsilon
    # Update to binary cross entropy loss
    target = tf.convert_to_tensor(target_y, dtype='float32')
    output = tf.convert_to_tensor(predicted_y, dtype='float32')
    epsilon_ = tf.constant(epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    bceloss = target * tf.math.log(output + epsilon())
    bceloss += (1-target) * tf.math.log(1 - output + epsilon())
    wbceloss = backend.mean(-bceloss * LW) 
    return wbceloss

def bce_builtin(target_y, predicted_y, label_weights=None):
    label_weights = tf.constant(label_weights, dtype='float32')
    label_weights = backend.mean(label_weights)
    #print(label_weights.numpy())
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return bce(target_y, predicted_y).numpy()*label_weights

ypred = [[0.2, 0.6, 0.01, 0.95],
        [0.7, 0.3, 0.9, 0.8]]
ytrue = [[0, 0, 0, 1],
        [0, 0, 1, 1]]
LW = [10,10,0.005,0.005]
print(f"Built-In Loss: {bce_builtin(ytrue,ypred,label_weights=LW)}")
print(f"Custom Loss: {wbce_loss(ytrue,ypred,label_weights=LW)}")