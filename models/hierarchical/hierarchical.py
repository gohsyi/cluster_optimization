import numpy as np

from tqdm import tqdm

from baselines.a2c.a2c import Model
from baselines.a2c.utils import discount_with_dones

from models.hierarchical.predictor import Predictor


class Hierarchical(object):
    """
    Hierarchical model
    """

    def __init__(self, g_ob_size, g_act_size, g_latents, g_lr, g_ac,
                 l_ob_size, l_act_size, l_latents, l_lr, l_ac):

        self.name = 'hierarchical'

        self.global_model = Model('global', g_ob_size, g_act_size, g_latents, g_lr, g_ac)
        self.local_model = Model('local', l_ob_size, l_act_size, l_latents, l_lr, l_ac)
        self.predictor = Predictor('hierarchical')

    def step(self, ob):
        action, value = self.global_model.step(ob)
        return action

    def value(self, ob):
        action, value = self.global_model.step(ob)
        return value


def learn_hierarchical(env, batch_size, total_epoches, gamma,
                       g_ob_size, g_act_size, g_latents, g_lr, g_ac,
                       l_ob_size, l_act_size, l_latents, l_lr, l_ac):
    """
    Learn a hierarchical model
    """

    model = Hierarchical(g_ob_size, g_act_size, g_latents, g_lr, g_ac,
                         l_ob_size, l_act_size, l_latents, l_lr, l_ac)

    mb_obs, mb_acts, mb_rews, mb_vals, mb_dones = [], [], [], [], []

    tqdm.write('training hierarchical model')

    for ep in tqdm(range(total_epoches)):
        obs = env.reset(model.local_model, model.predictor)  # initial observation
        done = False

        while not done:
            action = model.step(obs)  # RL choose action based on observation
            value = model.value(obs)
            action = int(action)
            value = int(value)
            mb_obs.append(obs)
            mb_acts.append(action)
            mb_vals.append(value)

            obs_, reward, done, info = env.step(action)
            mb_rews.append(reward)
            mb_dones.append(done)

            obs = obs_

            if ep % batch_size == 0:
                mb_rews = discount_with_dones(mb_rews, mb_dones, gamma)
                model.global_model.train(
                    np.array(mb_obs), np.array(mb_acts),
                    np.array(mb_rews), np.array(mb_vals)
                )
                mb_obs, mb_acts, mb_rews, mb_vals, mb_dones = [], [], [], [], []

    return model
