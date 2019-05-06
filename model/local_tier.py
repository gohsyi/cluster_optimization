# -*- coding: utf-8 -*-
"""
Created on Sun May  5 00:21:40 2019

@author: yeshe
"""
from model.predictor import Predictor
from model.base import BaseModel
from model.dqn import DeepQNetwork

import numpy as np


class LocalTier(BaseModel):
    def __init__(self, args):
        super(LocalTier, self).__init__(args)
        self.n_features = 100
        self.n_actions = 100
        self.w = .5

        self.T_on = args.T_on
        self.T_off = args.T_off
        self.P0 = args.P0
        self.P100 = args.P100

        self.predictor = Predictor(
            hdims=self.hid_size,
            memory_size=self.memory_size,
            lr=self.lr,
            ac_fn=self.ac_fn,
            opt_fn=self.opt_fn
        )
        self.QL = DeepQNetwork(
            n_actions=100,
            n_features=1 + self.n_features,
            learning_rate=self.lr,
            replace_target_iter=self.replace_target_iter,
            memory_size=self.memory_size,
            batch_size=self.batch_size,
        )
        self.index = -1
        self.work = []
        self.interval = []
        self.laststart = 0
        self.state = 0
        self.obs_ = [0, 0]
        self.cur_time = 0
        self.cpu = 0
        self.doing = []
        self.doingcost = []
        self.awake = 0

    def getpower(self, cpu):
        return self.P0 + (self.P100 - self.P0) * (2 * cpu - cpu ** (1.4))

    def getJob(self, start_time, last_time, cost):
        if (self.laststart == 0):
            self.laststart = start_time
        else:
            self.interval.append(start_time - self.laststart)
            self.laststart = start_time
        if (len(self.interval) > 1):
            tmp = []
            if (len(self.interval) > 36):
                for i in range(35):
                    tmp.append(self.interval[i + len(self.interval) - 36])
            else:
                for i in range(len(self.interval) - 1):
                    tmp.append(self.interval[i])
            self.predictor.train(tmp, self.interval[len(self.interval) - 1])
        if (self.state == 0):
            self.work.append([last_time, cost])
            if (self.awake > start_time + self.T_on):
                self.awake = start_time + self.T_off
            obs = [0]
            for i in range(self.n_features):
                obs.append(0)
            rew = -(1 - self.w) * len(self.work)
            act = []
            for i in range(self.n_actions):
                act.append(0)
            self.QL.store_transition(obs, act, rew, self.obs_)
            self.obs = obs
        elif (self.state == 1 and self.cpu == 0):
            obs = [self.getpower(0)]
            for i in range(self.n_features):
                obs.append(0)
            rew = -self.w * self.getpower(0)
            act = []
            for i in range(self.n_actions):
                act.append(0)
            self.QL.store_transition(obs, act, rew, self.obs_)
            self.obs = obs
        elif (100 - self.cpu < cost):
            self.work.append([last_time, cost])
        else:
            self.doing.append(start_time + last_time)
            self.doingcost.append(cost)
            self.cpu = self.cpu + cost
        self.cur_time = start_time

    def getnext(self):
        if (self.awake > 0):
            return self.awake
        elif (len(self.doing) > 0):
            return np.min(self.doing)
        else:
            return -1

    def done(self):
        if (self.awake > 0):
            self.cur_time = self.awake
            self.state = 1
            self.awake = -1
            while (len(self.work) != 0 and 100 - self.cpu > self.work[0][1]):
                self.cpu = self.cpu + self.work[0][1]
                self.doing.append(self.work[0][0] + self.cur_time)
                self.doingcost.append(self.work[0][1])
                del self.work[0]
        elif (len(self.doing) > 0):
            i = int(np.argmin(self.doing))
            self.cur_time = self.doing[i]
            self.cpu = self.cpu - self.doingcost[i]
            del self.doing[i]
            del self.doingcost[i]
            while (len(self.work) != 0 and 100 - self.cpu > self.work[0][1]):
                self.cpu = self.cpu + self.work[0][1]
                self.doing.append(self.work[0][0] + self.cur_time)
                self.doingcost.append(self.work[0][1])
                del self.work[0]
            if (len(self.doing) == 0 and len(self.work) == 0):
                if (len(self.interval) == 0):
                    t = 0
                else:
                    tmp = []
                    if (len(self.interval) > 35):
                        for i in range(35):
                            tmp.append(self.interval[i + len(self.interval) - 35])
                    else:
                        for i in range(len(self.interval)):
                            tmp.append(self.interval[i])
                    t = self.predictor.predict(tmp)
                t = int(t)
                if (t > self.n_features):
                    t = self.n_features
                obs = [self.getpower(0)]
                for i in range(t - 1):
                    obs.append(0)
                obs.append(1)
                for i in range(self.n_features - t):
                    obs.append(0)
                rew = -self.w * self.getpower(0)
                act = self.QL.choose_action(obs)
                self.QL.store_transition(obs, act, rew, self.obs_)
                self.QL.learn()
                self.obs = obs
                self.awake = self.cur_time + np.argmax(act)
                self.state = 0
