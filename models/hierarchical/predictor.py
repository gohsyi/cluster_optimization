import numpy as np
import tensorflow as tf

from baselines.common import get_session


class Predictor():
    """
    This model is basically hard coded,
    because parameter tuning has little effects on the preformance
    """

    def __init__(self, name):
        self.X = tf.placeholder(tf.float32, [None, None, 1])
        self.Y = tf.placeholder(tf.float32, [None, 1])

        with tf.variable_scope(name):
            lstm_cells = []
            for i, hdim in enumerate([64, 64]):
                lstm_cell_ = tf.nn.rnn_cell.LSTMCell(hdim, activation=tf.nn.relu)
                lstm_cells.append(lstm_cell_)

            cell_ = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
            init_state = cell_.zero_state(batch_size=1, dtype=tf.float32)

            outputs_, states_ = tf.nn.dynamic_rnn(cell_, self.X, initial_state=init_state)
            pred_ = tf.layers.dense(
                states_[-1][-1], 1,
                activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            self.pred_ = tf.clip_by_value(pred_, tf.constant(0.), tf.constant(10000.))

        self.loss_ = tf.reduce_mean(tf.squared_difference(self.pred_, self.Y))
        self.opt_ = tf.train.AdamOptimizer(1e-4).minimize(self.loss_)

        self.sess = get_session()
        self.sess.run(tf.global_variables_initializer())

    def predict(self, test_data):
        return self.sess.run(self.pred_, feed_dict={self.X: np.reshape(test_data, [1, -1, 1])})

    def train(self, train_data, train_label):
        pred, loss, _ = self.sess.run([self.pred_, self.loss_, self.opt_], feed_dict={
            self.X: train_data,
            self.Y: train_label
        })
