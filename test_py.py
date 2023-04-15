import random

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')


def live_plotter(y_num, size):
    plt.ion()
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111)
    lines = ax.plot([i for i in range(size)], [0] * size, '-o', alpha=0.8)
    plt.ylabel('Y Label')
    plt.title('Title: {}'.format('1'))
    plt.show()

    def f(ys):
        lines[0].set_ydata(ys[0])
        # lines[0].set_xdata(x)  # if this is done, need to set xlim as well
        if np.min(ys[0]) < lines[0].axes.get_ylim()[0] or np.max(ys[0]) > lines[0].axes.get_ylim()[1]:
            plt.ylim([np.min(ys[0]) - np.std(ys[0]), np.max(ys[0]) + np.std(ys[0])])
        plt.pause(.1)

    return f


y_values = [[0] * 100]
plotter = live_plotter(1, 100)

while True:
    for idx in range(len(y_values)):
        y_values[idx].append(random.randint(-100, 100))
        y_values[idx] = y_values[idx][1:]

    plotter(y_values)
