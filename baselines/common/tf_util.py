import os
import tensorflow as tf


def get_activation(ac_fn):
    """
    get activation function
    :param ac_fn: name of activation function,
                  eg: 'relu', 'sigmoid', 'elu', 'tanh'
    :return: corresponding activation function
    """

    if ac_fn == 'tanh':
        return tf.nn.tanh
    elif ac_fn == 'relu':
        return tf.nn.relu
    elif ac_fn == 'sigmoid':
        return tf.nn.sigmoid
    elif ac_fn == 'elu':
        return tf.nn.elu
    else:
        raise ValueError


def get_optimizer(opt_fn):
    """
    get optimizer function
    :param opt_fn: name of optimizer method
                   eg: 'sgd', 'adam', 'adagrad'
    :return:
    """
    if opt_fn == 'gd':
        return tf.train.GradientDescentOptimizer
    elif opt_fn == 'adam':
        return tf.train.AdamOptimizer
    elif opt_fn == 'adagrad':
        return tf.train.AdagradOptimizer
    elif opt_fn == 'rms':
        return tf.train.RMSPropOptimizer
    elif opt_fn == 'momentum':
        return tf.train.MomentumOptimizer
    else:
        raise ValueError


def get_session():
    """
    get default session of tensorflow,
    allowing soft placement for the better use of GPU

    :return: default tf session
    """

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    return tf.Session(config=config)



# ================================================================
# Saving variables
# ================================================================

def load_state(fname, sess=None):
    sess = sess or get_session()
    saver = tf.train.Saver()
    saver.restore(sess, fname)


def save_state(fname, sess=None):
    sess = sess or get_session()
    dirname = os.path.dirname(fname)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    saver = tf.train.Saver()
    saver.save(sess, fname)


def save_variables(save_path, variables=None, sess=None):
    import joblib
    sess = sess or get_session()
    variables = variables or tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    ps = sess.run(variables)
    save_dict = {v.name: value for v, value in zip(variables, ps)}
    dirname = os.path.dirname(save_path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(save_dict, save_path)


def load_variables(load_path, variables=None, sess=None):
    import joblib
    sess = sess or get_session()
    variables = variables or tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    loaded_params = joblib.load(os.path.expanduser(load_path))
    restores = []
    if isinstance(loaded_params, list):
        assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
        for d, v in zip(loaded_params, variables):
            restores.append(v.assign(d))
    else:
        for v in variables:
            restores.append(v.assign(loaded_params[v.name]))

    sess.run(restores)


def sample(logits):
    cliped_logits = tf.clip_by_value(logits, tf.constant(1e-4), tf.constant(1-(1e-4)))
    return tf.squeeze(tf.multinomial(tf.log(cliped_logits), 1), -1)

def sample_k(logits, k):
    """
    sample the largest k logits
    :param logits: logits before softmax
    :param k: number of samples
    :return:
        tensor, indices of the largest k logits, represented with onehot
        the shape is (batch_size x act_size)
    """

    act_size = logits.shape[-1]
    noise = tf.random_uniform(tf.shape(logits))
    _, indices = tf.nn.top_k(logits - tf.log(-tf.log(noise)), k=k)

    return tf.reduce_sum(tf.one_hot(indices, act_size), 1)
