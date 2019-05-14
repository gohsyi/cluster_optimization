import numpy as np
from env import Env
from model.base import BaseModel
from model.dqn import DeepQNetwork
import matplotlib.pyplot as plt
plt.style.use('ggplot')


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

    def run(self):
        ### hierarchical DRL
        # done = False
        # info = None
        # ep = 0
        # obs = self.env.reset(False)  # initial observation
        # while not done:
        #     act = self.global_model.choose_action(obs)  # RL choose action based on observation
        #     obs_, rew, done, info = self.env.step(act)  # RL take action and get next observation and reward
        #     self.global_model.store_transition(obs, act, rew, obs_)
        #     if ep > 200 and ep % 5 == 0:
        #         self.global_model.learn()
        #     obs = obs_  # swap observation
        #     ep += 1
        # rl_power_usage, rl_latency = info

        ### only local server
        done = False
        info = None
        obs = self.env.reset(True)  # initial observation
        while not done:
            m_cpu = (100, 0)
            for i, m in enumerate(self.env.machines):
                if m.cpu() < m_cpu[0]:
                    m_cpu = (m.cpu(), i)
            obs_, rew, done, info = self.env.step(m_cpu[1])
        rl_power_usage, rl_latency = info

        # done = False
        # info = None
        # ep = 0
        # # obs = self.env.reset(False)  # initial observation
        # self.env.latency = []
        # self.env.power_usage = []
        # for m in self.env.machines:
        #     m.power_usage = 0
        # while not done:
        #     act = self.global_model.choose_action(obs)  # RL choose action based on observation
        #     obs_, rew, done, info = self.env.step(act)  # RL take action and get next observation and reward
        #     # test
        #     # self.global_model.store_transition(obs, act, rew, obs_)
        #     # if ep > 200 and ep % 5 == 0:
        #     #     self.global_model.learn()
        #     obs = obs_  # swap observation
        #     ep += 1
        # rl_power_usage, rl_latency = info

        ### random dispatch
        done = False
        info = None
        self.env.reset(False)
        while not done:
            act = np.random.randint(self.n_servers)
            obs_, rew, done, info = self.env.step(act)
        r_power_usage, r_latency = info

        ### round robin
        done = False
        info = None
        act = 0
        self.env.reset(False)
        while not done:
            obs_, rew, done, info = self.env.step(act)
            act = (act + 1) % self.n_servers
        rr_power_usage, rr_latency = info

        ### greedy
        done = False
        info = None
        self.env.reset(False)
        while not done:
            m_cpu = (100, 0)
            for i, m in enumerate(self.env.machines):
                if m.cpu() < m_cpu[0]:
                    m_cpu = (m.cpu(), i)
            obs_, rew, done, info = self.env.step(m_cpu[1])
        g_power_usage, g_latency = info

        ### plot power usage
        plt.title('Power Usage')
        plt.plot(r_power_usage, c='g', label='Random')
        plt.plot(rr_power_usage, c='b', label='Round Robin')
        plt.plot(g_power_usage, c='r', label='Greedy')
        plt.plot(rl_power_usage, c='c', label='Hierarchical DRL')
        plt.xlabel('Number of Jobs')
        plt.ylabel('Energy Usage (kWh)')
        plt.legend()
        plt.savefig('figures/power_usage')
        plt.cla()

        ### plot latency
        plt.title('Latency')
        plt.plot(r_latency, c='g', label='Random')
        plt.plot(rr_latency, c='b', label='Round Robin')
        plt.plot(g_latency, c='r', label='Greedy')
        plt.plot(rl_latency, c='c', label='Hierarchical DRL')
        plt.xlabel('Number of Jobs')
        plt.ylabel('Accumulated Job latency')
        plt.legend()
        plt.savefig('figures/latency')
        plt.cla()
