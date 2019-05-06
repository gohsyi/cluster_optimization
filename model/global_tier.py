import numpy as np
from env import Env
from model.base import BaseModel
from model.dqn import DeepQNetwork
from model.predictor import Predictor
from model.local_tier import LocalTier


class Model(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.env = Env(args.n_servers)
        self.global_model = DeepQNetwork(
            n_actions=self.n_servers,
            n_features=self.n_servers * self.n_resources + self.n_resources + 1
        )
        self.local_model = LocalTier(args)
        self.predict_model = Predictor(
            hdims=self.hid_size,
            memory_size=self.memory_size,
            lr=self.lr,
            ac_fn=self.ac_fn,
            opt_fn=self.opt_fn
        )

    def train(self):
        obs = self.env.reset()  # initial observation

        for ep in range(self.max_epoches):
            obs[-1] = np.squeeze(self.predict_model.predict(obs[:-1], obs[-1]))
            act = self.global_model.choose_action(obs)  # RL choose action based on observation
            obs_, rew = self.env.step(act)  # RL take action and get next observation and reward
            self.global_model.store_transition(obs, act, rew, obs_)

            if ep > 200 and ep % 5 == 0:
                self.global_model.learn()

            obs = obs_  # swap observation
