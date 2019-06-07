import numpy as np

from baselines.a2c.a2c import Model
from models.hierarchical.predictor import Predictor


class Local(object):
    """
    Only local tier
    """

    def __init__(self, n_servers, l_ob_size, l_act_size, l_latents, l_lr, l_ac):
        self.n_servers = n_servers
        self.name = 'local'

        self.local_model = Model('local_only', l_ob_size, l_act_size, l_latents, l_lr, l_ac)
        self.predictor = Predictor('local_lstm')

    def step(self, obs):
        m_cpu = (100, [])
        for i in range(self.n_servers):
            cpu = obs[i << 1]
            if cpu < m_cpu[0]:
                m_cpu = (cpu, [i])
            elif cpu == m_cpu[0]:
                m_cpu[1].append(i)
        return np.random.choice(m_cpu[1])


def learn_local(env, total_epoches, n_servers,
                l_ob_size, l_act_size, l_latents, l_lr, l_ac):
    """
    Learn a hierarchical model
    """

    model = Local(n_servers, l_ob_size, l_act_size, l_latents, l_lr, l_ac)

    for ep in range(total_epoches):
        obs = env.reset(model.local_model, model.predictor)  # initial observation
        done = False

        while not done:
            action = int(model.step(obs))  # RL choose action based on observation
            obs_, reward, done, info = env.step(action)
            obs = obs_

    return model
