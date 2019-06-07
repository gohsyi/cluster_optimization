import numpy as np
from common import get_logger


class Stochastic(object):
    def __init__(self, act_size):
        self.name = 'stochastic'
        self.act_size = act_size
        self.logger = get_logger(self.name)

        self.local_model = None
        self.predictor = None

    def step(self, obs):
        return np.random.randint(self.act_size)
