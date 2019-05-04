import os
import pandas as pd


class Env(object):
    def __init__(self):
        # data paths
        machine_meta_path = os.path.join('data', 'machine_meta.csv')
        machine_usage_path = os.path.join('data', 'machine_usage.csv')
        container_meta_path = os.path.join('data', 'container_meta.csv')
        container_usage_path = os.path.join('data', 'container_usage.csv')
        batch_task_path = os.path.join('data', 'batch_task.csv')
        batch_instance_path = os.path.join('data', 'batch_instance.csv')

        # data columns
        machine_meta_cols = [
            'machine_id',  # uid of machine
            'time_stamp',  # time stamp, in second
            'failure_domain_1',  # one level of container failure domain
            'failure_domain_2',  # another level of container failure domain
            'cpu_num',  # number of cpu on a machine
            'mem_size',  # normalized memory size. [0, 100]
            'status',  # status of a machine
        ]
        machine_usage_cols = [
            'machine_id',
            'time_stamp',
            'cpu_util_percent',
            'mem_util_percent',
            'mem_gps',
            'mkpi',
            'net_in',
            'net_out',
            'disk_io_percent',
        ]
        container_meta_cols = [
            'container_id',
            'machine_id',
            'time_stamp',
            'app_du',
            'status',
            'cpu_request',
            'cpu_limit',
            'mem_size',
        ]

        # read csv into DataFrames
        machine_meta = pd.read_csv(machine_meta_path, columns=machine_meta_cols)
