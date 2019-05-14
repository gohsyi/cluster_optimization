import tensorflow as tf


class BaseModel(object):
    def __init__(self, args):
        self.n_servers = args.n_servers
        self.n_resources = args.n_resources
        self.coef_totalpower = args.w1
        self.coef_numbervms = args.w2
        self.coef_reliobj = args.w3
        self.max_epoches = args.max_epoches
        self.batch_size = args.batch_size
        self.replace_target_iter = args.replace_target_iter
        self.memory_size = args.memory_size

        self.a2c_hidsizes = list(map(int, args.a2c_hidsizes.split(',')))
        self.lstm_hidsizes = list(map(int, args.lstm_hidsizes.split(',')))

        self.dqn_lr = args.dqn_lr
        self.a2c_lr = args.a2c_lr
        self.lstm_lr = args.lstm_lr

        if args.ac_fn == 'tanh':
            self.ac_fn = tf.nn.tanh
        elif args.ac_fn == 'relu':
            self.ac_fn = tf.nn.relu
        elif args.ac_fn == 'sigmoid':
            self.ac_fn = tf.nn.sigmoid
        elif args.ac_fn == 'elu':
            self.ac_fn = tf.nn.elu
        else:
            raise ValueError

        if args.opt_fn == 'sgd':
            self.opt_fn = tf.train.GradientDescentOptimizer
        elif args.opt_fn == 'adam':
            self.opt_fn = tf.train.AdamOptimizer
        elif args.opt_fn == 'adagrad':
            self.opt_fn = tf.train.AdagradOptimizer
        else:
            raise NotImplementedError
