import numpy as np
import tensorflow as tf
from model.base import BaseModel
from util import getLogger


class Predictor(BaseModel):
    def __init__(self, args):
        super(Predictor, self).__init__(args)
        self.X = tf.placeholder(tf.float32, [None, None, 1])
        self.Y = tf.placeholder(tf.float32, [None, 1])

        self.lr = args.lstm_lr
        self.hidsizes = self.lstm_hidsizes

        self.logger = getLogger('logs', 'predictor')

        lstm_cells = []
        for i, hdim in enumerate(self.hidsizes):
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
        self.pred_ = tf.clip_by_value(pred_, tf.constant(0), tf.constant(10000))
        self.loss_ = tf.reduce_mean(tf.squared_difference(self.pred_, self.Y))
        self.opt_ = self.opt_fn(self.lr).minimize(self.loss_)

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def predict(self, test_data):
        return self.sess.run(self.pred_, feed_dict={self.X: np.reshape(test_data, [1, -1, 1])})

    def train(self, train_data, train_label):
        pred, loss, _ = self.sess.run([self.pred_, self.loss_, self.opt_], feed_dict={
            self.X: train_data,
            self.Y: train_label
        })
        self.logger.info('loss:{}\tpred:{}\tlabel:{}'.format(loss, np.mean(pred), np.mean(train_label)))
