import os
import pandas as pd
from queue import Queue


class Env(object):
    def __init__(self, n_servers):
        self.n_servers = n_servers
        self.p_idle = 87
        self.p_full = 145
        self.t_on = 30
        self.t_off = 30

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
        self.cur_task = 0

    def loadcsv(self):
        #  read csv into DataFrames
        self.machine_meta = pd.read_csv(self.machine_meta_path, header=None, names=self.machine_meta_cols)
        self.machine_meta = self.machine_meta[self.machine_meta['time_stamp'] == 0]
        self.machine_meta = self.machine_meta[['machine_id', 'cpu_num', 'mem_size']]

        self.batch_task = pd.read_csv(self.batch_task_path, header=None, names=self.batch_task_cols)
        self.batch_task = self.batch_task[self.batch_task['status'] == 'Terminated'].sort_values(by='start_time')

        self.n_machines = self.n_servers
        self.machines = [
            Machine(self.machine_meta[i]['cpu_num'],
                    self.machine_meta[i]['mem_size']
            ) for i in range(self.n_machines)
        ]

    def sleep(self, id, time):
        if time > 0:
            self.machines[id].sleep(self.cur_time + time + self.t_off + self.t_on)

    def reset(self):
        self.cur_task = 0
        cur_time = self.batch_task[self.cur_task]['start_time']
        cur_task = Task(
            self.batch_task[self.cur_task]['start_time'],
            self.batch_task[self.cur_task]['end_time'],
            self.batch_task[self.cur_task]['plan_cpu'],
            self.batch_task[self.cur_task]['plan_mem']
        )
        states = []
        for m in self.machines:
            states.extend(m.process(cur_time))
        states.extend([cur_task.plan_cpu, cur_task.plan_mem, cur_task.end_time - cur_task.start_time])

        return states

    def step(self, action):
        self.cur_time = self.batch_task[self.cur_task]['start_time']
        cur_task = Task(
            self.batch_task[self.cur_task]['start_time'],
            self.batch_task[self.cur_task]['end_time'],
            self.batch_task[self.cur_task]['plan_cpu'],
            self.batch_task[self.cur_task]['plan_mem']
        )
        self.cur_task += 1
        nxt_task = Task(
            self.batch_task[self.cur_task]['start_time'],
            self.batch_task[self.cur_task]['end_time'],
            self.batch_task[self.cur_task]['plan_cpu'],
            self.batch_task[self.cur_task]['plan_mem']
        )
        self.machines[action].add_task(cur_task)

        states = []
        for m in self.machines:
            states.extend(m.process(self.cur_time))
        states.extend([nxt_task.plan_cpu, nxt_task.plan_mem, nxt_task.end_time - nxt_task.start_time])

        return states


class Task(object):
    def __init__(self, start_time, end_time, plan_cpu, plan_mem):
        self.start_time = start_time
        self.end_time = end_time
        self.plan_cpu = plan_cpu
        self.plan_mem = plan_mem

    def start(self, actual_start_time):
        self.actual_start_time = actual_start_time

    def done(self, cur_time):
        return cur_time >= self.end_time - self.start_time + self.actual_start_time


class Machine(object):
    def __init__(self, cpu_num, mem_size):
        self.waiting_queue = Queue()
        self.running_queue = Queue()
        self.cpu_num = cpu_num
        self.mem_size = mem_size
        self.cur_task = None
        self.sleep_util = 0

    def add_task(self, task):
        self.waiting_queue.put(task)

    def sleep(self, sleep_util):
        self.sleep_util = sleep_util

    def process(self, cur_time):
        running_queue = Queue()
        waiting_queue = Queue()

        if cur_time >= self.sleep_util:
            while not self.running_queue.empty():
                task = self.running_queue.get()
                if task.done(cur_time):
                    self.cpu_num += task.plan_cpu
                    self.mem_size += task.plan_mem
                else:
                    running_queue.put(task)

            while not self.waiting_queue.empty():
                task = self.running_queue.get()
                if task.plan_cpu <= self.cpu_num and task.plan_mem <= self.mem_size:
                    self.cpu_num -= task.plan_cpu
                    self.mem_size -= task.plan_mem
                    task.start(cur_time)
                    running_queue.put(task)
                else:
                    waiting_queue.put(task)

            self.running_queue = running_queue
            self.waiting_queue = waiting_queue

        return [self.cpu_num, self.mem_size]


if __name__ == '__main__':
    env = Env(100)
