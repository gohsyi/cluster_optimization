import heapq
import numpy as np
import tensorflow as tf
from collections import deque
from model.predictor import Predictor
from model.dqn import DeepQNetwork


class Machine(object):
    def __init__(self, args, cpu_num, mem_size):
        self.P_0 = args.P_0
        self.P_100 = args.P_100
        self.T_on = args.T_on
        self.T_off = args.T_off

        self.pending_queue = deque()
        self.running_queue = []

        self.cpu_num = cpu_num
        self.mem_size = mem_size
        self.cpu_idle = cpu_num
        self.mem_empty = mem_size

        self.cur_task = None
        self.awake_time = 0
        self.intervals = deque(maxlen=35 + 1)
        self.state = 1
        self.T_on = 30
        self.T_off = 30
        self.w = 0.5

        self.n_features = 100
        self.n_actions = 100

        self.last_arrive_time = 0
        self.last_obs = None
        self.last_act = None
        self.rew = 0
        self.cur_time = 0

        with tf.variable_scope('local_model', reuse=tf.AUTO_REUSE):
            self.predictor = Predictor(args)
            self.QL = DeepQNetwork(
                n_actions=100,
                n_features=1 + self.n_features
            )

    def cpu(self):
        return 1 - self.cpu_idle / self.cpu_num

    def add_task(self, task):
        self.pending_queue.append(task)

        ### train predictor
        self.last_arrive_time = task.arrive_time
        if self.last_arrive_time != 0:
            self.intervals.append(task.arrive_time - self.last_arrive_time)
        if len(self.intervals) > 1:
            self.predictor.train(self.intervals[:-1], self.intervals[-1])

        ### a task arrives and interrupts the sleeping
        if self.state == 0:
            self.pending_queue.append(task)
            if (self.awake_time > task.arrive_time + self.T_on):
                self.awake_time = task.arrive_time + self.T_on
            self.rew -= (1 - self.w)

        ### a task comes and the server has already awaken
        elif self.state == 1 and self.cpu() == 0:
            self.rew -= self.w * self.P_0 * (task.arrive_time - self.cur_time)

    def process(self, cur_time):
        if self.awake_time > cur_time:
            self.cur_time = cur_time
            return
        self.cur_time = max(self.cur_time, self.awake_time)

        while True:
            while len(self.pending_queue) > 0:
                task = self.pending_queue[0]
                if task.plan_cpu <= self.cpu_idle and task.plan_mem <= self.mem_empty:
                    task.start(cur_time)
                    self.pending_queue.popleft()
                    self.cpu_idle -= task.plan_cpu
                    self.mem_empty -= task.plan_mem
                    heapq.heappush(self.running_queue, (self.cur_time + task.last_time, task))
                else:
                    break

            end_time, task = self.running_queue[0]
            if end_time <= cur_time:
                heapq.heappop(self.running_queue)
                self.cpu_idle += task.plan_cpu
                self.mem_empty += task.plan_mem
                self.cur_time = end_time
            else:
                break

        self.cur_time = cur_time

        if len(self.running_queue) == 0 and len(self.pending_queue) == 0:
            pred = min(self.n_features, int(self.predictor.predict(self.intervals)))
            obs = [self.P_0, np.eye(self.n_features)[pred]]
            act = self.QL.choose_action(obs)
            self.QL.store_transition(self.obs, self.act, self.rew, obs)
            self.QL.learn()
            self.obs = obs
            self.act = act
            self.awake = self.cur_time + act
            self.state = 0
