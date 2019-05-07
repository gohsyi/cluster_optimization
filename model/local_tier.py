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
        self.state = 'waken'
        self.T_on = 30
        self.T_off = 30
        self.w = 0.5

        self.n_features = 100
        self.n_actions = 100

        self.last_arrive_time = 0
        self.obs = None
        self.act = None
        self.rew = 0
        self.cur_time = 0

        with tf.variable_scope('local_model', reuse=tf.AUTO_REUSE):
            self.predictor = Predictor(args)
            self.QL = DeepQNetwork(
                name='local',
                n_actions=100,
                n_features=1 + self.n_features,
                learning_rate=args.dqn_lr
            )

    def cpu(self):
        return 1 - self.cpu_idle / self.cpu_num

    def add_task(self, task):
        self.pending_queue.append(task)

        ### train predictor
        if self.last_arrive_time != 0 and task.arrive_time != self.last_arrive_time:
            self.intervals.append(task.arrive_time - self.last_arrive_time)
        self.last_arrive_time = task.arrive_time
        if len(self.intervals) > 1:
            self.predictor.train(
                np.reshape(list(self.intervals)[:-1], [1, -1, 1]),
                np.reshape(list(self.intervals)[-1], [1, 1])
            )

        ### a task arrives and interrupts the sleeping
        if self.state == 'sleeping':
            self.pending_queue.append(task)
            if (self.awake_time > task.arrive_time + self.T_on):
                self.awake_time = task.arrive_time + self.T_on
            self.rew -= (1 - self.w)

        ### a task comes and the server has already awaken
        elif self.state == 'awake' and self.cpu() == 0:
            self.rew -= self.w * self.P_0 * (task.arrive_time - self.cur_time)

    def process_running_queue(self, cur_time):
        end_time, task = self.running_queue[0]
        if end_time <= cur_time:
            heapq.heappop(self.running_queue)
            self.cpu_idle += task.plan_cpu
            self.mem_empty += task.plan_mem
            self.cur_time = end_time
            return False
        return True

    def process_pending_queue(self, cur_time):
        task = self.pending_queue[0]
        self.cur_time = max(self.cur_time, task.arrive_time)
        if task.plan_cpu <= self.cpu_idle and task.plan_mem <= self.mem_empty:
            task.start(self.cur_time)
            self.pending_queue.popleft()
            self.cpu_idle -= task.plan_cpu
            self.mem_empty -= task.plan_mem
            heapq.heappush(self.running_queue, (self.cur_time + task.last_time, task))
            return False
        return True

    def process(self, cur_time):
        if self.cur_time == 0:  # the first process
            self.cur_time = cur_time
            return
        if self.awake_time > cur_time:  # not waken at cur_time
            self.cur_time = cur_time
            return
        if self.awake_time > self.cur_time:  # jump to self.awake_time
            self.cur_time = self.awake_time
            self.state = 'waken'

        timeout, stuck = False, False

        while True:
            if len(self.running_queue) > 0 and (
                    stuck or len(self.pending_queue) == 0 or self.running_queue[0][0] <= self.pending_queue[0].arrive_time):
                timeout = self.process_running_queue(cur_time)
                if timeout:
                    break

            elif len(self.pending_queue) > 0 and (
                    len(self.running_queue) == 0 or self.pending_queue[0].arrive_time < self.running_queue[0][0]):
                stuck = self.process_pending_queue(cur_time)

            elif self.state != 'waken':  # both running queue and pending queue are empty
                if len(self.intervals) > 0:
                    pred = min(self.n_features, int(self.predictor.predict(self.intervals)))
                else:
                    pred = 0
                obs = np.concatenate([np.array([self.P_0]) / 10, np.eye(self.n_features)[pred]], axis=-1)
                act = self.QL.choose_action(obs)
                if self.obs is not None:
                    self.QL.store_transition(self.obs, self.act, self.rew, obs)
                    self.QL.learn()
                self.obs = obs
                self.act = act
                self.rew = 0

                if act > 0:
                    self.awake_time = self.cur_time + act + self.T_on + self.T_off
                    self.state = 'sleeping'

                break

        self.cur_time = cur_time
