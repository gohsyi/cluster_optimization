import numpy as np
from env import Env
from model.base import BaseModel
from model.dqn import DeepQNetwork


class Model(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.lr = args.dqn_lr
        self.env = Env(args)
        self.global_model = DeepQNetwork(
            name='global',
            learning_rate=self.lr,
            n_actions=self.n_servers,
            n_features=self.n_servers * self.n_resources + self.n_resources + 1
        )

    def train(self):
        obs = self.env.reset()  # initial observation

        for ep in range(self.max_epoches):
            act = self.global_model.choose_action(obs)  # RL choose action based on observation
            obs_, rew, cur_time = self.env.step(act)  # RL take action and get next observation and reward
            self.global_model.store_transition(obs, act, rew, obs_)

            if ep > 200 and ep % 5 == 0:
                self.global_model.learn()

            obs = obs_  # swap observation
