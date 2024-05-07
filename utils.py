import gc
import tensorflow as tf
from tensorflow import keras
from keras.metrics import Metric
from keras import backend as B
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.callbacks import ModelCheckpoint

# Need to update this to come from other file
NUM_CLASSES = 25

class GarbageMan(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

# Custom Label Metric for Prediction matrix
class PerLabelMetric(Metric):
    def __init__(self,name='per_label_metric', num_labels=NUM_CLASSES, threshold=None,**kwargs):
        super(PerLabelMetric, self).__init__(name=name,**kwargs)
        self.num_labels = num_labels
        self.t = threshold
        self.tp = self.add_weight(name='tp', shape=(self.num_labels), initializer='zeros')
        self.tn = self.add_weight(name='tn', shape=(self.num_labels), initializer='zeros')
        self.fp = self.add_weight(name='fp', shape=(self.num_labels), initializer='zeros')
        self.fn = self.add_weight(name='fn', shape=(self.num_labels), initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Custom logic to compute the metric for each label
        for i in range(self.num_labels):
            y_true_label = y_true[:, i]
            y_pred_label = y_pred[:, i]
            ypos = tf.cast(tf.math.greater_equal(y_pred_label, self.t), tf.float32)
            yneg = tf.cast(tf.math.less(y_pred_label, self.t), tf.float32)
            tp = B.sum(y_true_label * ypos, axis=0)
            fp = B.sum((1 - y_true_label) * ypos, axis=0)
            tn = B.sum((1 - y_true_label) * yneg, axis=0)
            fn = B.sum(y_true_label * yneg, axis=0)

            self.tp[i].assign(self.tp[i] + tp)
            self.fp[i].assign(self.fp[i] + fp)
            self.tn[i].assign(self.tn[i] + tn)
            self.fn[i].assign(self.fn[i] + fn)

    def result(self):
        tp = self.tp
        tn = self.tn
        fp = self.fp
        fn = self.fn
        acc = (tp + tn) / (tp + tn + fp + fn)
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = (2 * tp) / ((2 * tp) + fp + fn)
        return tp.numpy(), tn.numpy(), fp.numpy(), fn.numpy(), acc.numpy(), p.numpy(), r.numpy(), f1.numpy()

    def reset_state(self):
        # Reset the state of the metric
        B.batch_set_value([(v, 0) for v in self.variables])

# Custom Weighted Loss Function
def wbce_loss(target_y, predicted_y, label_weights=None):
    from keras.src import backend, backend_config
    epsilon = backend_config.epsilon
    target = tf.convert_to_tensor(target_y, dtype='float32')
    output = tf.convert_to_tensor(predicted_y, dtype='float32')
    epsilon_ = tf.constant(epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    bceloss = target * tf.math.log(output + epsilon())
    bceloss += (1-target) * tf.math.log(1 - output + epsilon())
    if label_weights != None:
        lw=np.array(list(label_weights.items()))
        lw = lw[:,1]
        wbceloss = backend.mean(-bceloss * lw) 
    else:
        wbceloss = backend.mean(-bceloss) 
    return wbceloss