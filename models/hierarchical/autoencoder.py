import numpy as np
import tensorflow as tf


class AutoEncoder(object):
    def __init__(self, sess, feature_dim, hidsize, max_epoches, lr, ac_fn, opt_fn, training_data):
        self.feature_dim = feature_dim
        self.sess = sess
        self.max_epoches = max_epoches
        self.lr = lr
        self.ac_fn = ac_fn
        self.opt_fn = opt_fn

        self.X = tf.placeholder(tf.float32, [None, feature_dim], 'feature_dim')

        # build encoder
        hidlayer = self.X
        for i, hdim in enumerate(hidsize):
            hidlayer = tf.layers.dense(
                self.X, hdim,
                activation=ac_fn,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
        self.encoder_ = hidlayer

        # build decoder
        for i, hdim in enumerate(reversed(hidsize)):
            hidlayer = tf.layers.dense(
                self.X, hdim,
                activation=ac_fn,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
        self.decoder_ = hidlayer

        self.loss_ = tf.reduce_mean(tf.squared_difference(self.X, self.decoder_))
        self.opt_ = self.opt_fn(self.lr).minimize(self.loss_)

        self.sess.run(tf.global_variables_initializer())

    def train(self, x):
        avg_loss = []
        for ep in range(self.max_epoches):
            loss, _ = self.sess.run([self.loss_, self.opt_], feed_dict={self.X: x})
            avg_loss.append(loss)
            if ep % 100 == 0:
                print('ep:{}\tloss:{}'.format(ep, np.mean(avg_loss)))

    def encode(self, x):
        return self.sess.run(self.encoder_, feed_dict={self.X: x})
