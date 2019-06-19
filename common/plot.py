import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import numpy as np


class SmoothPlot():
    def __init__(self, smooth_rate=0.9, linewidth=1.0):
        self.smooth_rate = smooth_rate
        self.linewidth = linewidth
        self.colors = ['r', 'b', 'c', 'g', 'm', 'y', 'k', 'w']

    def plot(self, data, save_path, title=None, label=None):
        if type(data) == list and type(label) == list:
            for d, l, c in zip(data, label, self.colors):
                plt.plot(d, c=c, alpha=0.2, linewidth=1.0)
                plt.plot(self.smooth_momentum(d), label=l, c=c, linewidth=self.linewidth)
        else:
            plt.plot(data, c='r', alpha=0.2, linewidth=1.0)
            plt.plot(self.smooth_momentum(data), label=label, c='r', linewidth=self.linewidth)

        if title:
            plt.title(title)

        if label:
            plt.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.cla()

    def smooth_momentum(self, arr):
        ret = [arr[0]]
        for i in range(1, len(arr)):
            ret.append(ret[i - 1] * self.smooth_rate + arr[i] * (1-self.smooth_rate))
        return ret

    def smooth_average(self, arr):
        ret = [arr[0]]
        for i in range(1, len(arr), 10):
            ret.append(float(np.mean(arr[i:i + 10])))
        return ret
