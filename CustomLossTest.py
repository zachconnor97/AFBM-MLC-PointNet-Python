import tensorflow as tf
from keras.src import backend_config
epsilon = backend_config.epsilon

def loss(target_y, predicted_y, label_weights=None):
    # Update to binary cross entropy loss
    #target_y = tf.cast(target_y, dtype=tf.float32)  # Assuming float32 is the desired data type
    #print("Target shape:", target_y.shape)
    #print("Predicted shape:", predicted_y.shape)
    target = tf.convert_to_tensor(target_y)
    output = tf.convert_to_tensor(predicted_y)
    print(target)
    print(output)
    #bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # Label weights not working
    epsilon_ = tf.constant(epsilon(), output.dtype.base_dtype)
    print(epsilon_)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    # multiply label weights by binary output
    # collapse label weights, multiply loss by weighting. Still becomes same weight?
    # maybe need to loop along labels to get the weights in
    wbceloss = target * tf.math.log(output + epsilon())
    wbceloss += (1-target) * tf.math.log(1 - predicted_y + epsilon())
    return -wbceloss #bce(target_y, predicted_y).numpy() #bce(target_y, predicted_y, label_weight=label_weights).numpy()

ypred = [0.2,0.6,0.01,0.95]
ytrue = [1,0,0,1]
LW = [0.01,1,0.5,0.5]
print(f"Loss: {loss(ytrue,ypred,label_weights=LW)}")