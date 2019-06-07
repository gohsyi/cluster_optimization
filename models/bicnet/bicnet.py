import numpy as np
import tensorflow as tf
from common.util import get_logger


class BiCNet(BaseException):
    def __init__(self, name, n_actions, n_features, hidsizes, lr, ac_fn, vf_coef=0.1, seed=0):
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.logger = get_logger('logs', name)

        self.n_actions = n_actions
        self.n_features = n_features
        self.hidsizes = hidsizes
        self.ac_fn = ac_fn
        self.lr = lr
        self.vf_coef = vf_coef

        self._build_net()

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        self.X = tf.placeholder(tf.float32, [None, self.n_features], name='state')  # input State
        self.ADV = tf.placeholder(tf.float32, [None, ], name='advantage')  # input Advantage
        self.R = tf.placeholder(tf.float32, [None, ], name='reward')  # input Reward
        self.A = tf.placeholder(tf.int32, [None, ], name='action')  # input Action

        with tf.variable_scope('actor'):
            lstm_fw_cells = []
            for hdim in self.hidsizes:
                lstm_fw_cells.append(tf.nn.rnn_cell.LSTMCell(hdim, activation=self.ac_fn))
            lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_fw_cells)

            lstm_bw_cells = []
            for hdim in self.hidsizes:
                lstm_bw_cells.append(tf.nn.rnn_cell.LSTMCell(hdim, activation=self.ac_fn))
            lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_bw_cells)

            outputs, state_fw, state_bw = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, self.X, dtype=tf.float32)

            self.latent = [tf.concat(outputs, axis=-1)]
            self.logits = self.dense(self.latent, self.n_actions)  # to compute loss
            self.pi = tf.nn.softmax(self.logits)
            self.act = tf.multinomial(tf.log(self.pi), 1)

        with tf.variable_scope('critic'):
            vf_latent = self.X
            for hsize in self.hidsizes:
                vf_latent = self.dense(vf_latent, hsize, self.ac_fn)
            self.vf = tf.squeeze(self.dense(vf_latent, 1), -1)

        with tf.variable_scope('loss'):
            self.neglogp = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=tf.clip_by_value(tf.log(self.pi), tf.constant(-1e4), tf.constant(1e4)),
                labels=self.A
            )
            self.pg_loss = tf.reduce_mean(self.neglogp * self.ADV)
            self.vf_loss = tf.reduce_mean(tf.squared_difference(self.vf, self.R))
            self.loss = self.pg_loss + self.vf_coef * self.vf_loss

        with tf.variable_scope('train'):
            self.trainer = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, obs):
        return self.sess.run(self.act, feed_dict={self.X: [obs]})

    def dense(self, inputs, units, ac_fn=None):
        return tf.layers.dense(
            inputs, units,
            activation=ac_fn,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.5)
        )
