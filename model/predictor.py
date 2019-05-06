import tensorflow as tf


class Predictor(object):
    def __init__(self, hdims, memory_size, lr, ac_fn, opt_fn):
        self.ac_fn = ac_fn
        self.opt_fn = opt_fn
        self.lr = lr
        self.X = tf.placeholder(tf.float32, [None, memory_size, 1])
        self.Y = tf.placeholder(tf.int32, [None, 1])

        lstm_cells = []
        for i, hdim in enumerate(hdims):
            lstm_cell_ = tf.nn.rnn_cell.LSTMCell(
                hdim,
                activation=self.ac_fn,
            )
            lstm_cells.append(lstm_cell_)
        cell_ = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        outputs_, states_ = tf.nn.dynamic_rnn(cell_, self.X)
        self.pred_ = tf.layers.dense(
            tf.stack(outputs_, 1), 1,
            kernel_initializer=tf.truncated_normal_initializer
        )

        self.loss_ = tf.squared_difference(self.pred_, self.Y)
        self.opt_ = self.opt_fn(self.lr).minimize(self.loss_)

    def predict(self, test_data):
        with tf.Session() as sess:
            return sess.run(self.pred_, feed_dict={self.X: test_data})

    def train(self, train_data, train_label):
        with tf.Session() as sess:
            pred, loss, _ = sess.run([self.pred_, self.loss_, self.opt_], feed_dict={
                self.X: train_data,
                self.Y: train_label
            })
            print('loss:{}'.format(loss))
