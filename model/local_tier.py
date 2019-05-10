import heapq
import numpy as np
import tensorflow as tf
from collections import deque
from model.predictor import Predictor
from model.dqn import DeepQNetwork


class Machine(object):
    def __init__(self, args, cpu_num, mem_size, machine_id, is_baseline):
        self.machine_id = machine_id
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

        self.cur_time = 0
        self.awake_time = 0
        self.intervals = deque(maxlen=35 + 1)
        self.state = 'waken'  # waken, active, sleeping
        self.w = 0.5

        self.n_features = 100
        self.n_actions = 100

        self.last_arrive_time = 0
        self.power_usage = 0
        self.obs = None
        self.act = None
        self.rew = 0

        self.is_baseline = is_baseline

        if not self.is_baseline:
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

        if not self.is_baseline:
            self.train_predictor(task)
            if self.state == 'sleeping':
                self.try_to_wake_up(task)
            elif self.state == 'active' and self.cpu_idle == self.cpu_num:
                self.rew -= self.w * self.P_0 * (task.arrive_time - self.cur_time)

        self.process_pending_queue()

    """
    Process running queue, return whether we should process running queue or not
    We should process running queue first if it's not empty and any of these conditions holds:
    1. Pending queue is empty
    2. The first task in pending queue cannot be executed for the lack of resources (cpu or memory)
    3. The first task in pending queue arrives after any task in the running queue finishes
    """
    def process_running_queue(self, cur_time):
        if self.is_empty(self.running_queue):
            return False
        if self.running_queue[0].end_time > cur_time:
            return False

        if self.is_empty(self.pending_queue) or \
            not self.enough_resource(self.pending_queue[0]) or \
            self.running_queue[0].end_time <= self.pending_queue[0].arrive_time:

            task = heapq.heappop(self.running_queue)
            self.state = 'active'
            self.cpu_idle += task.plan_cpu
            self.mem_empty += task.plan_mem

            # update power usage
            self.power_usage += self.calc_power(task.end_time)

            self.cur_time = task.end_time

            return True

        return False

    """
    We should process pending queue first if it's not empty and 
    the server has enough resources (cpu and memory) for the first task in the pending queue to run and
    any of these following conditions holds:
    1. Running queue is empty
    2. The first task in the pending queue arrives before all tasks in the running queue finishes
    """
    def process_pending_queue(self):
        if self.is_empty(self.pending_queue):
            return False
        if not self.enough_resource(self.pending_queue[0]):
            return False

        if self.is_empty(self.running_queue) or \
            self.pending_queue[0].arrive_time < self.running_queue[0].end_time:

            task = self.pending_queue.popleft()
            task.start(self.cur_time)
            self.cpu_idle -= task.plan_cpu
            self.mem_empty -= task.plan_mem
            heapq.heappush(self.running_queue, task)

            return True

        return False

    """ 
    keep running simulation until current time 
    """
    def process(self, cur_time):
        if self.cur_time == 0:  ## the first time, no task has come before
            self.cur_time = cur_time
            return
        if self.awake_time > cur_time:  ## will not be waken at cur_time
            self.cur_time = cur_time
            return
        if self.awake_time > self.cur_time:  ## jump to self.awake_time
            self.cur_time = self.awake_time
            self.state = 'waken'

        while self.process_pending_queue() or self.process_running_queue(cur_time):
            pass

        if not self.is_baseline:
            if self.state != 'awake' and \
                    self.is_empty(self.pending_queue) and \
                    self.is_empty(self.running_queue):
                if len(self.intervals) > 0:
                    pred = min(self.n_features-1, int(self.predictor.predict(self.intervals)) // 10)
                else:
                    pred = 0
                obs = np.concatenate([np.array([self.P_0]), np.eye(self.n_features)[pred]], axis=-1)
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

        self.power_usage += self.calc_power(cur_time)
        self.cur_time = cur_time

    def enough_resource(self, task):
        return task.plan_cpu <= self.cpu_idle and task.plan_mem <= self.mem_empty

    def is_empty(self, queue):
        return len(queue) == 0

    def calc_power(self, cur_time):
        if self.state == 'sleeping':
            return 0
        else:
            cpu = self.cpu()
            return (self.P_0 + (self.P_100 - self.P_0) * (2*cpu - cpu**1.4)) * (cur_time - self.cur_time)

    def finish_jobs(self):
        cur_time = self.cur_time
        while not self.is_empty(self.pending_queue) or not self.is_empty(self.running_queue):
            while self.process_pending_queue() or self.process_running_queue(cur_time + 1):
                pass
            cur_time += 1

    def train_predictor(self, task):
        if self.last_arrive_time != 0 and task.arrive_time != self.last_arrive_time:
            self.intervals.append(task.arrive_time - self.last_arrive_time)
        self.last_arrive_time = task.arrive_time
        if len(self.intervals) > 1:
            self.predictor.train(
                np.reshape(list(self.intervals)[:-1], [1, -1, 1]),
                np.reshape(list(self.intervals)[-1], [1, 1])
            )

    def try_to_wake_up(self, task):
        if (self.awake_time > task.arrive_time + self.T_on):
            self.awake_time = task.arrive_time + self.T_on
        self.rew -= (1 - self.w)
