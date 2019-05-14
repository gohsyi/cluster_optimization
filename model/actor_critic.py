import numpy as np
import tensorflow as tf
from util import getLogger


class ActorCritic(object):
    def __init__(self, name, n_actions, n_features, hidsizes, lr, ac_fn,
                 vf_coef=0.1, memory_size=1000, batch_size=128, seed=0):
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.logger = getLogger('logs', name)

        self.n_actions = n_actions
        self.n_features = n_features
        self.hidsizes = hidsizes
        self.ac_fn = ac_fn
        self.lr = lr
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.vf_coef = vf_coef

        # initialize zero memory [state, action, advantage]
        self.memory = np.zeros((self.memory_size, n_features + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        self.S = tf.placeholder(tf.float32, [None, self.n_features], name='state')  # input State
        self.ADV = tf.placeholder(tf.float32, [None, ], name='advantage')  # input Advantage
        self.R = tf.placeholder(tf.float32, [None, ], name='reward')  # input Reward
        self.A = tf.placeholder(tf.int32, [None, ], name='action')  # input Action

        with tf.variable_scope('actor'):
            pi_latent = self.S
            for hsize in self.hidsizes:
                pi_latent = self.dense(pi_latent, hsize, self.ac_fn)
            self.logits = self.dense(pi_latent, self.n_actions)  # to compute loss
            self.pi = tf.nn.softmax(self.logits)
            self.act = tf.multinomial(tf.log(self.pi), 1)

        with tf.variable_scope('critic'):
            vf_latent = self.S
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

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r]))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, obs):
        return self.sess.run(self.act, feed_dict={self.S: [obs]})

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        v = self.sess.run(self.vf, feed_dict={self.S: batch_memory[:, :self.n_features]})

        _, cost = self.sess.run([self.trainer, self.loss], feed_dict={
            self.S: batch_memory[:, :self.n_features],
            self.A: batch_memory[:, self.n_features],
            self.R: batch_memory[:, self.n_features + 1],
            self.ADV: batch_memory[:, self.n_features + 1] - v,
        })

        self.logger.info('loss:{}\trew:{}'.format(cost, np.mean(batch_memory[:, self.n_features + 1])))

    def dense(self, inputs, units, ac_fn=None):
        return tf.layers.dense(
            inputs, units,
            activation=ac_fn,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.5)
        )
