from common import get_logger


class RoundRobin(object):
    def __init__(self, act_size):
        self.name = 'round_robin'
        self.act_size = act_size
        self.action = 0
        self.logger = get_logger(self.name)

        self.local_model = None
        self.predictor = None

    def step(self, obs):
        self.action = (self.action + 1) % self.act_size
        return self.action
