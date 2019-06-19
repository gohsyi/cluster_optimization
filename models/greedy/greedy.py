import numpy as np
from common import get_logger


class Greedy(object):
    def __init__(self, act_size, n_servers):
        self.name = 'greedy'
        self.n_servers = n_servers
        self.act_size = act_size
        self.logger = get_logger(self.name)

        self.local_model = None
        self.predictor = None

    def step(self, obs):
        m_cpu = (100, [])
        for i in range(self.n_servers):
            cpu = obs[i << 1]
            if cpu < m_cpu[0]:
                m_cpu = (cpu, [i])
            elif cpu == m_cpu[0]:
                m_cpu[1].append(i)
        return np.random.choice(m_cpu[1])
