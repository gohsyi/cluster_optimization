import random
import numpy as np
import tensorflow as tf


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
