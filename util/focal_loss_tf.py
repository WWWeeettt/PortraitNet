import tensorflow as tf
from tensorflow.python.ops import array_ops

def focal_loss(prediction_tensor, target_tensor, alpha=None, gamma=2):
    if len(prediction_tensor.shape) > 2:
        prediction_tensor = tf.reshape(prediction_tensor, [-1, prediction_tensor.shape[-1]])
    target_tensor = tf.reshape(target_tensor, [-1, 1])

    logpt = tf.nn.log_softmax(prediction_tensor)
    logpt = tf.gather(logpt, target_tensor, 1)
    logpt = tf.reshape(logpt, [-1])
    pt = tf.math.exp(logpt)

    if alpha is not None:
        pass
    
    loss = -1 * (1 - pt) ** gamma * logpt

    return tf.reduce_mean(loss)