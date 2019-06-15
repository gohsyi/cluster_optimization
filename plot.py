import os
import numpy as np
import argparse

from common.plot import SmoothPlot


parser = argparse.ArgumentParser()
parser.add_argument('-smooth_rate', type=float, default=0)
parser.add_argument('-line_width', type=float, default=1)
args = parser.parse_args()

plt = SmoothPlot(args.smooth_rate, args.line_width)


def plot():
    model_list = ['stochastic', 'round_robin', 'greedy', 'hierarchical', 'local']

    for root, dirs, files in os.walk('logs'):
        for dir in dirs:
            latency_list = [np.loadtxt(os.path.join('logs', dir, f'{model}_latency.txt')) for model in model_list]
            power_list = [np.loadtxt(os.path.join('logs', dir, f'{model}_power.txt')) for model in model_list]

            plt.plot(latency_list, label=model_list, save_path=os.path.join('logs', dir, 'latency.png'), title='latency')
            plt.plot(power_list, label=model_list, save_path=os.path.join('logs', dir, 'power.png'), title='power usage')


if __name__ == '__main__':
    plot()
