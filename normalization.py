

from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow.contrib.layers import batch_norm
### batch normalization
def batch_norm(inputs, decay=0.9, is_training=True, epsilon=1e-6):
    """

    :param inputs:  [batch, length, width, channels]
    :param is_training:
    :param eplison:
    :return:
    """
    pop_mean = tf.Variable(tf.zeros(inputs.shape[-1]), trainable=False, name="pop_mean")
    pop_var = tf.Variable(tf.ones(inputs.shape[-1]), trainable=False, name="pop_variance")

    def update_mean_and_var():
        axes = list(range(inputs.shape.ndims))
        batch_mean, batch_var = tf.nn.moments(inputs, axes=axes)
        moving_average_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1-decay))
        # 也可用 assign_moving_average(pop_mean, batch_mean, decay)
        moving_average_var = tf.assign(pop_var, pop_var * decay + batch_var * (1-decay))
        # 也可用 assign_moving_average(pop_var, batch_var, decay)
        with tf.control_dependencies([moving_average_mean, moving_average_var]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, variance = tf.cond(tf.equal(is_training, True), update_mean_and_var,
                        lambda: (pop_mean, pop_var))
    beta = tf.Variable(initial_value=tf.zeros(inputs.get_shape()[-1]), name="shift")
    gamma = tf.Variable(initial_value=tf.ones(inputs.get_shape()[-1]), name="scale")
    return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)



import tensorflow as tf
def layer_norm_mine(inputs, epsilon=1e-12, center=True, scale=True):
    """
    inputs: [batch, sequence_len, hidden_size] or [batch, hidden_size]
    """
    inputs_shape = inputs.shape
    inputs_rank = inputs_shape.ndims
    params_shape = inputs_shape[-1:]
    beta, gamma = None, None
    if center:
        beta = tf.get_variable(
            name="beta",
            shape=params_shape,
            initializer=tf.zeros_initializer(),
            trainable=True
        )
    if scale:
        gamma = tf.get_variable(
            name="gamma",
            shape=params_shape,
            initializer=tf.ones_initializer(),
            trainable=True
        )
    norm_axes = list(range(1, inputs_rank))
    mean, variance = tf.nn.moments(inputs, norm_axes, keep_dims=True)      # [batch]
    inv = tf.rsqrt(variance + epsilon)
    inv *= gamma
    return inputs*inv + ((beta-mean)*inv if beta is not None else - mean * inv)


if __name__ == "__main__":

    batch = 60
    hidden_size = 64
    whh = tf.random_normal(shape=[batch, hidden_size], mean=5.0, stddev=10.0)
    whh_norm = layer_norm_mine(whh)
