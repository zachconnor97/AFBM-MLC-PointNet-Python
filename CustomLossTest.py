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

def pc_loss(tt, tg, label_weights=None):
    # find average of x, y, and z coords
    xt_mn = backend.mean(tt[:,0])
    yt_mn = backend.mean(tt[:,1])
    zt_mn = backend.mean(tt[:,2])
    xg_mn = backend.mean(tg[:,0])
    yg_mn = backend.mean(tg[:,1])
    zg_mn = backend.mean(tg[:,2])
    # find second moment of inertia tensors M2_g M2_t

    # Invert M2_t, calculate R_inv = M2_g, M2_t_inv
    # loss = abs((R_inv)^-1 - IDM)
    pc_loss = 1 #place holder
    return pc_loss

def loss(target_y, predicted_y):
    # cloud loss
    xt_mn = predicted_y
    # find average of x, y, and z coords
    # find second moment of inertia tensors M2_g M2_t
    # Invert M2_t, calculate R_inv = M2_g, M2_t_inv
    # loss = abs((R_inv)^-1 - IDM)

    target_y = tf.cast(target_y, dtype=tf.float32)  # Assuming float32 is the desired data type
    #print("Target shape:", target_y.shape)
    #print("Predicted shape:", predicted_y.shape)
    l = tf.reduce_mean(tf.square(target_y - predicted_y))
    print(type(l))
    print(l)
    return l
"""
ypred = [[0.2, 0.6, 0.01, 0.95],
        [0.7, 0.3, 0.9, 0.8]]
ytrue = [[0, 0, 0, 1],
        [0, 0, 1, 1]]
LW = [10,10,0.005,0.005]
#print(f"Built-In Loss: {bce_builtin(ytrue,ypred,label_weights=LW)}")
#print(f"Custom Loss: {wbce_loss(ytrue,ypred,label_weights=LW)}")
t = tf.constant([0.2])
pos = tf.math.greater_equal(tf.constant(ypred),t)
neg = tf.math.less(tf.constant(ypred),t)
pos = tf.cast(pos, tf.float32)
neg = tf.cast(neg, tf.float32)
ytrue = tf.cast(ytrue, tf.float32)

print(f"Positives at Threshold: {t.numpy()} : {pos}")
print(f"Negatives at Threshold: {t.numpy()} : {neg}")

print(f"Y True: {ytrue}")
"""
pc_gen = [[0.5, 0.2, 0.4],
          [0.2, 0.1, 0.3],
          [0.9, 0.2, 0.7],
          [0.8, 0.5, 0.9],
          [0.9, 0.1, 0.0]]

pc_true = [[0.2, 0.1, 0.5],
          [0.2, 0.1, 0.3],
          [0.9, 0.2, 0.7],
          [0.7, 0.59, 0.2],
          [0.8, 0.11, 0.23]]

pc_gen = tf.constant(pc_gen, tf.float32)
pc_true = tf.constant(pc_true, tf.float32)
print(pc_gen[:,0].numpy())
