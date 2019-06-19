import tensorflow as tf


def dense(inputs, units, name=None, activation=None):
    """
    construct a fully-connected layer

    :param inputs: inputs of fc
    :param units: number of outputs
    :param activation: activation function

    :return: corresponding created FC layer
    """

    return tf.layers.dense(
        inputs, units,
        # name=name,
        activation=activation,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.5)
    )


def mlp(x, latents, activation=None):
    """
    construct a multi-layer perception

    :param x: input
    :param latents: latent sizes
    :param activation: activation function
    :return: corresponding created MLP
    """

    last_latent = x
    for i, hdim in enumerate(latents):
        last_latent = dense(last_latent, hdim, 'layer%i' % i, activation)
    return last_latent


def lstm(x, latents, activation=None, dropout=1.):
    """
    construct an lstm network

    :param x: input
    :param latents: latent sizes
    :param activation: activation function (default None)
    :param dropout: dropout rate (default 1. no dropout)
    :return: created lstm network
    """

    lstm_cells = []
    for hdim in latents:
        lstm_cell = tf.nn.rnn_cell.LSTMCell(hdim, activation=activation)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell,
            input_keep_prob=dropout,
            output_keep_prob=dropout
        )
        lstm_cells.append(lstm_cell)

    cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
    initial_state = cell.zero_state(tf.shape(x)[0], tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state)

    return final_state[-1].h
