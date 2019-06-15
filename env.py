import os
import heapq
import pandas as pd
import numpy as np

from collections import deque

from common.util import get_logger
from common.argparser import args


class Env(object):
    def __init__(self):
        self.logger = get_logger('env')
        self.log = self.logger.info
        self.log(str(args))

        self.P_0 = args.P_0
        self.P_100 = args.P_100
        self.T_on = args.T_on
        self.T_off = args.T_off
        self.n_servers = args.n_servers

        self.w1 = args.w1
        self.w2 = args.w2
        self.w3 = args.w3

        #  data paths
        self.machine_meta_path = os.path.join('data', 'machine_meta.csv')
        self.machine_usage_path = os.path.join('data', 'machine_usage.csv')
        self.container_meta_path = os.path.join('data', 'container_meta.csv')
        self.container_usage_path = os.path.join('data', 'container_usage.csv')
        self.batch_task_path = os.path.join('data', 'batch_task.csv')
        self.batch_instance_path = os.path.join('data', 'batch_instance.csv')

        #  data columns
        self.machine_meta_cols = [
            'machine_id',  # uid of machine
            'time_stamp',  # time stamp, in second
            'failure_domain_1',  # one level of container failure domain
            'failure_domain_2',  # another level of container failure domain
            'cpu_num',  # number of cpu on a machine
            'mem_size',  # normalized memory size. [0, 100]
            'status',  # status of a machine
        ]
        self.machine_usage_cols = [
            'machine_id',  # uid of machine
            'time_stamp',  # time stamp, in second
            'cpu_util_percent',  # [0, 100]
            'mem_util_percent',  # [0, 100]
            'mem_gps',  # normalized memory bandwidth, [0, 100]
            'mkpi',  # cache miss per thousand instruction
            'net_in',  # normarlized in coming network traffic, [0, 100]
            'net_out',  # normarlized out going network traffic, [0, 100]
            'disk_io_percent',  # [0, 100], abnormal values are of -1 or 101 |
        ]
        self.container_meta_cols = [
            'container_id',  # uid of a container
            'machine_id',  # uid of container's host machine  
            'time_stamp',  # 
            'app_du',  # containers with same app_du belong to same application group
            'status',  # 
            'cpu_request',  # 100 is 1 core 
            'cpu_limit',  # 100 is 1 core 
            'mem_size',  # normarlized memory, [0, 100]
        ]
        self.container_usage_cols = [
            'container_id',  # uid of a container
            'machine_id',  # uid of container's host machine  
            'time_stamp',  # 
            'cpu_util_percent',
            'mem_util_percent',
            'cpi',
            'mem_gps',  # normalized memory bandwidth, [0, 100]
            'mpki',
            'net_in',  # normarlized in coming network traffic, [0, 100]
            'net_out',  # normarlized out going network traffic, [0, 100]
            'disk_io_percent'  # [0, 100], abnormal values are of -1 or 101
        ]
        self.batch_task_cols = [
            'task_name',  # task name. unique within a job
            'instance_num',  # number of instances  
            'job_name',  # job name
            'task_type',  # task type
            'status',  # task status
            'start_time',  # start time of the task
            'end_time',  # end of time the task
            'plan_cpu',  # number of cpu needed by the task, 100 is 1 core
            'plan_mem'  # normalized memorty size, [0, 100]
        ]
        self.batch_instance_cols = [
            'instance_name',  # instance name of the instance
            'task_name',  # task name. unique within a job
            'instance_num',  # number of instances  
            'job_name',  # job name
            'task_type',  # task type
            'status',  # task status
            'start_time',  # start time of the task
            'end_time',  # end of time the task
            'machine_id',  # uid of host machine of the instance
            'seq_no'  # sequence number of this instance
            'total_seq_no',  # total sequence number of this instance
            'cpu_avg',  # average cpu used by the instance, 100 is 1 core
            'cpu_max',  # average memory used by the instance (normalized)
            'mem_avg',  # max cpu used by the instance, 100 is 1 core
            'mem_max',  # max memory used by the instance (normalized, [0, 100])
        ]

        self.loadcsv()
        self.cur = 0
        self.power_usage = []
        self.latency = []

    def loadcsv(self):
        # self.logger.info('loading csv file ...')

        #  read csv into DataFrames
        self.machine_meta = pd.read_csv(self.machine_meta_path, header=None, names=self.machine_meta_cols)
        self.machine_meta = self.machine_meta[self.machine_meta['time_stamp'] == 0]
        self.machine_meta = self.machine_meta[['machine_id', 'cpu_num', 'mem_size']]

        self.batch_task = pd.read_csv(self.batch_task_path, header=None, names=self.batch_task_cols)
        self.batch_task = self.batch_task[self.batch_task['status'] == 'Terminated']
        self.batch_task = self.batch_task[self.batch_task['plan_cpu'] <= 100]  # will stuck the pending queue
        self.batch_task = self.batch_task[self.batch_task['start_time'] > 85000]
        self.batch_task = self.batch_task.sort_values(by='start_time')
        # self.batch_task['end_time'] -= pd.DataFrame.min(self.batch_task['start_time'])
        # self.batch_task['start_time'] -= pd.DataFrame.min(self.batch_task['start_time'])

        self.n_machines = self.n_servers
        self.n_tasks = args.n_tasks or self.batch_task.shape[0]

        self.tasks = [ Task(
            self.batch_task.iloc[i]['task_name'],
            self.batch_task.iloc[i]['start_time'],
            self.batch_task.iloc[i]['end_time'],
            self.batch_task.iloc[i]['plan_cpu'],
            self.batch_task.iloc[i]['plan_mem']
        ) for i in range(self.n_tasks) ]

    def clean(self):
        self.cur = 0
        self.power_usage = []
        self.latency = []

        for m in self.machines:
            m.power_usage = 0

        return self.get_states(self.tasks[self.cur])

    def reset(self, local_model, predictor):
        self.cur = 0
        self.power_usage = []
        self.latency = []

        self.machines = [ Machine(
            100, 100,
            self.machine_meta.iloc[i]['machine_id'],
            local_model,
            predictor,
        ) for i in range(self.n_machines) ]

        return self.get_states(self.tasks[self.cur])

    def step(self, action):
        self.cur_time = self.batch_task.iloc[self.cur]['start_time']
        cur_task = self.tasks[self.cur]

        done = False
        self.cur += 1
        if self.cur == self.n_tasks:
            self.latency = [t.start_time - t.arrive_time for t in self.tasks]
            for i in range(1, len(self.latency)):
                self.latency[i] = self.latency[i] + self.latency[i - 1]
            # for m in self.machines:
            #     m.finish_jobs()
            done = True
            self.cur = 0

        nxt_task = self.tasks[self.cur]
        # self.logger.info('dispatch task {} to machine {}'.format(
        #     cur_task.name,
        #     self.machines[action].machine_id)
        # )

        ### simulate to current time
        for m in self.machines:
            m.process(self.cur_time)

        self.power_usage.append(np.sum([m.power_usage for m in self.machines]))
        # self.logger.info('ep:{}\ttime:{}\tpower:{}\tlatency:{}'.format(
        #     self.cur,
        #     self.cur_time,
        #     self.power_usage[-1],
        #     np.sum([t.start_time - t.arrive_time for t in self.tasks])
        # ))

        self.machines[action].add_task(cur_task)

        return self.get_states(nxt_task), self.get_reward(), done, (self.power_usage, self.latency)

    def get_states(self, nxt_task):
        # self.logger.info('cpu:' + str([m.cpu_idle for m in self.machines]))
        # self.logger.info('mem:' + str([m.mem_empty for m in self.machines]))

        states = [m.cpu_idle for m in self.machines] + \
                 [m.mem_empty for m in self.machines] + \
                 [nxt_task.plan_cpu, nxt_task.plan_mem, nxt_task.last_time]
        return np.array(states)  # scale

    def get_reward(self):
        return -self.w1 * self.calc_total_power() + \
               -self.w2 * self.calc_number_vms() + \
               -self.w3 * self.calc_reli_obj()

    def calc_total_power(self):
        for m in self.machines:
            return self.P_0 + (self.P_100 - self.P_0) * (2 * m.cpu() - m.cpu()**(1.4))

    def calc_number_vms(self):
        return np.sum([len(m.pending_queue) for m in self.machines])

    def calc_reli_obj(self):
        return 0


class Task(object):
    def __init__(self, name, start_time, end_time, plan_cpu, plan_mem):
        self.name = name
        self.arrive_time = start_time
        self.last_time = end_time - start_time
        self.plan_cpu = plan_cpu
        self.plan_mem = plan_mem
        self.start_time = self.arrive_time

    def start(self, start_time):
        self.start_time = start_time
        self.end_time = start_time + self.last_time

    def done(self, cur_time):
        return cur_time >= self.start_time + self.last_time

    def __lt__(self, other):
        return self.start_time + self.last_time < other.start_time + other.last_time


class Machine():
    def __init__(self, cpu_num, mem_size, machine_id, local_model=None, predictor=None):
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
        self.n_features = args.l_obsize
        self.n_actions = args.l_actsize
        self.last_arrive_time = 0
        self.power_usage = 0
        self.observation = None
        self.action = None
        self.reward = 0
        self.model = local_model
        self.predictor = predictor

    def cpu(self):
        return 1 - self.cpu_idle / self.cpu_num

    def add_task(self, task):
        self.pending_queue.append(task)
        if self.predictor is not None:
            self.train_predictor(task)
        if self.model is not None:
            if self.state == 'sleeping':
                self.try_to_wake_up(task)
            elif self.state == 'active' and self.cpu_idle == self.cpu_num:
                self.reward -= self.w * self.P_0 * (task.arrive_time - self.cur_time)

        self.process_pending_queue()

    def process_running_queue(self, cur_time):
        """
        Process running queue, return whether we should process running queue or not
        We should process running queue first if it's not empty and any of these conditions holds:
        1. Pending queue is empty
        2. The first task in pending queue cannot be executed for the lack of resources (cpu or memory)
        3. The first task in pending queue arrives after any task in the running queue finishes
        """

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

    def process_pending_queue(self):
        """
        We should process pending queue first if it's not empty and
        the server has enough resources (cpu and memory) for the first task in the pending queue to run and
        any of these following conditions holds:
        1. Running queue is empty
        2. The first task in the pending queue arrives before all tasks in the running queue finishes
        """

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

    def process(self, cur_time):
        """
        keep running simulation until current time
        """

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

        if self.model is not None:
            if self.state != 'awake' and \
                    self.is_empty(self.pending_queue) and \
                    self.is_empty(self.running_queue):

                if len(self.intervals) > 0:
                    pred = min(self.n_features-1, int(self.predictor.predict(self.intervals)) // 10)
                else:
                    pred = 0

                obs = np.concatenate([np.array([self.P_0]), np.eye(self.n_features)[pred]], axis=-1)

                action, value = self.model.step(obs)
                action, value = int(action[0]), int(value[0])

                if self.observation is not None:
                    self.model.train(
                        np.array([self.observation]),
                        np.array([self.action]),
                        np.array([self.reward]),
                        np.array([self.value]))

                self.observation = obs
                self.action = action
                self.value = value
                self.reward = 0

                if action > 0:
                    self.awake_time = self.cur_time + action + self.T_on + self.T_off
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
        self.reward -= (1 - self.w)
