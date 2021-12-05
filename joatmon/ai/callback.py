import os

from joatmon.ai.core import CoreCallback


class CallbackList(CoreCallback):
    def __init__(self, callbacks):
        super(CallbackList, self).__init__()

        self.callbacks = callbacks

    def on_action_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_action_begin(*args, **kwargs)

    def on_action_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_action_end(*args, **kwargs)

    def on_agent_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_agent_begin(*args, **kwargs)

    def on_agent_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_agent_end(*args, **kwargs)

    def on_episode_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_episode_begin(*args, **kwargs)

    def on_episode_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_episode_end(*args, **kwargs)

    def on_replay_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_replay_begin(*args, **kwargs)

    def on_replay_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_replay_end(*args, **kwargs)


class Loader(CoreCallback):
    def __init__(self, model, run_path, interval):
        super().__init__()

        self.network = model
        self.run_path = run_path
        self.interval = interval

    def on_agent_begin(self, *args, **kwargs):
        # weights path should be run path
        self.network.load(self.run_path)
        self.network.save(self.run_path)

    def on_agent_end(self, *args, **kwargs):
        self.network.save(self.run_path)

    def on_episode_end(self, *args, **kwargs):
        if 'episode_number' in kwargs:
            if kwargs['episode_number'] % self.interval == 0:
                self.network.save(self.run_path)


class Renderer(CoreCallback):
    def __init__(self, environment):
        super().__init__()

        self.environment = environment

    def on_action_end(self, *args, **kwargs):
        self.environment.render()

    def on_episode_begin(self, *args, **kwargs):
        self.environment.render()


class TrainLogger(CoreCallback):
    def __init__(self, run_path, interval):
        super().__init__()

        self.run_path = run_path
        self.interval = interval

        if not os.path.exists(run_path):
            os.makedirs(run_path)

        self.agent_data_path = self.run_path + 'train-agent-data.csv'
        self.network_data_path = self.run_path + 'train-nn-data.csv'
        self.episode_end_message_raw = '\repisode {:05d} ended in {:04d} actions, total reward: {:+08.2f}'

    def on_agent_begin(self, *args, **kwargs):
        if 'agent_headers' in kwargs:
            with open(self.agent_data_path, 'w') as file:
                file.write(','.join(kwargs['agent_headers']) + '\n')

        if 'network_headers' in kwargs:
            with open(self.network_data_path, 'w') as file:
                file.write(','.join(kwargs['network_headers']) + '\n')

    def on_episode_begin(self, *args, **kwargs):
        pass

    def on_episode_end(self, *args, **kwargs):
        if 'episode_number' in kwargs and 'action_number' in kwargs and 'episode_reward' in kwargs:
            end = '\n' if kwargs['episode_number'] % self.interval == 0 else ''
            print(self.episode_end_message_raw.format(kwargs['episode_number'], kwargs['action_number'], kwargs['episode_reward']), end=end)
            with open(self.agent_data_path, 'a') as file:
                file.write(','.join(list(map(str, [kwargs['episode_number'], kwargs['action_number'], kwargs['episode_reward']]))) + '\n')

    def on_replay_end(self, *args, **kwargs):
        if 'loss' in kwargs:
            if not isinstance(kwargs['loss'], (list, tuple)):
                kwargs['loss'] = [kwargs['loss']]

            with open(self.network_data_path, 'a') as file:
                file.write(','.join(list(map(str, kwargs['loss']))) + '\n')


import sys
from PyQt5.QtWidgets import (
    QApplication,
    QStyleFactory
)
import matplotlib

matplotlib.use("Qt5Agg")

from PyQt5.QtWidgets import (
    QMainWindow,
    QFrame,
    QGridLayout
)
from PyQt5.QtCore import (
    QObject,
    pyqtSignal
)
from PyQt5.QtGui import QColor
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time


class CustomMainWindow(QMainWindow):
    def __init__(self):
        super(CustomMainWindow, self).__init__()
        # Define the geometry of the main window
        self.setGeometry(300, 300, 800, 400)
        self.setWindowTitle("my first window")
        # Create FRAME_A
        self.FRAME_A = QFrame(self)
        self.FRAME_A.setStyleSheet("QWidget { background-color: %s }" % QColor(210, 210, 235, 255).name())
        self.LAYOUT_A = QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.setCentralWidget(self.FRAME_A)
        # Place the matplotlib figure
        self.fig = CustomFigCanvas()
        self.LAYOUT_A.addWidget(self.fig, *(0, 1))
        self.show()
        return

    def callback(self, value):
        self.fig.add_data(value)
        return


class CustomFigCanvas(FigureCanvas, TimedAnimation):
    def __init__(self):
        print(matplotlib.__version__)

        self.x = [0]
        self.y = [0]

        # The window
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax1 = self.fig.add_subplot(111)
        # self.ax1 settings
        self.ax1.set_xlabel('time')
        self.ax1.set_ylabel('raw data')
        self.line = Line2D([], [], color='blue', marker='o')
        self.ax1.add_line(self.line)
        self.ax1.set_xlim(1, 12)
        self.ax1.set_ylim(-500, 500)
        self.ax1.get_xaxis().set_visible(False)
        self.ax1.get_yaxis().set_visible(False)
        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval=50, blit=True)
        return

    def new_frame_seq(self):
        return iter(range(20000))

    def _init_draw(self):
        # lines = [self.line1, self.line1_tail, self.line1_head]
        lines = [self.line]
        for line in lines:
            line.set_data([], [])
        return

    def add_data(self, value):
        self.x.append(self.x[-1] + 1 if len(self.x) > 0 else 1)
        self.y.append(value)
        return

    def _step(self, *args):
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            TimedAnimation._stop(self)
            pass
        return

    def _draw_frame(self, framedata):
        margin = 2
        # while len(self.data) > 0:
        #     self.y = np.roll(self.y, -1)
        #     self.y[-1] = self.data[0]
        #     del (self.data[0])

        # self.line.set_data(self.n[0: self.n.size - margin], self.y[0: self.n.size - margin])
        try:
            if len(self.y) // 40 > 0:
                import pandas as pd
                df = pd.DataFrame(self.y)
                y = df.rolling(window=len(self.y) // 40).mean().values
                y = y[~np.isnan(y)]
            else:
                y = self.y
            x = [x for x in range(len(y))]
            if len(self.x) > 0:
                x_max = max(x)
                x_min = min(x)
                y_max = max(y)
                y_min = min(y)
                x_lim_min, x_lim_max = self.ax1.get_xlim()
                y_lim_min, y_lim_max = self.ax1.get_ylim()
                self.ax1.set_xlim(x_min, x_max + 3)
                self.ax1.set_ylim(y_min - 3, y_max + 3)

                self.line.set_data(x, y)
                self._drawn_artists = [self.line]
        except Exception as ex:
            print(str(ex))
        return


class Communicate(QObject):
    data_signal = pyqtSignal(float)


def send_data(callback):
    # Setup the signal-slot mechanism.

    # Simulate some data
    n = np.linspace(0, 499, 500)
    y = 50 + 25 * (np.sin(n / 8.3)) + 10 * (np.sin(n / 7.5)) - 5 * (np.sin(n / 1.5))
    i = 0

    while True:
        if i > 499:
            i = 0
        time.sleep(0.1)
        i += 1


class TrainPlotter(CoreCallback):

    def __init__(self):
        super().__init__()

        import threading
        threading.Thread(target=self.draw).start()

        self.episode_numbers = []
        self.action_numbers = []
        self.episode_rewards = []
        self.losses = []
        self.gui = None
        self.src = None

    def draw(self):
        app = QApplication(sys.argv)
        QApplication.setStyle(QStyleFactory.create('Plastique'))
        self.gui = CustomMainWindow()

        self.src = Communicate()
        self.src.data_signal.connect(self.gui.callback)

        sys.exit(app.exec_())

    def on_episode_end(self, *args, **kwargs):
        if 'episode_number' in kwargs and 'action_number' in kwargs and 'episode_reward' in kwargs:
            episode_number = kwargs['episode_number']
            action_number = kwargs['action_number']
            episode_reward = kwargs['episode_reward']

            self.src.data_signal.emit(episode_reward)  # <- Here you emit a signal!

    def on_replay_end(self, *args, **kwargs):
        if 'loss' in kwargs:
            loss = kwargs['loss']


class ValidationLogger(CoreCallback):
    def __init__(self, run_path, interval):
        super().__init__()

        self.run_path = run_path
        self.interval = interval

        if not os.path.exists(run_path):
            os.makedirs(run_path)

        self.agent_data_path = self.run_path + 'test-agent-data.csv'
        self.episode_end_message_raw = '\repisode {:05d} ended in {:04d} actions, total reward: {:+08.2f}'

    def on_agent_begin(self, *args, **kwargs):
        with open(self.agent_data_path, 'w') as file:
            file.write('episode_number,action_number,episode_reward\n')

    def on_episode_begin(self, *args, **kwargs):
        pass

    def on_episode_end(self, *args, **kwargs):
        if 'episode_number' in kwargs and 'action_number' in kwargs and 'episode_reward' in kwargs:
            end = '\n' if kwargs['episode_number'] % self.interval == 0 else ''
            print(self.episode_end_message_raw.format(kwargs['episode_number'], kwargs['action_number'], kwargs['episode_reward']), end=end)
            with open(self.agent_data_path, 'a') as file:
                file.write(','.join(list(map(str, [kwargs['episode_number'], kwargs['action_number'], kwargs['episode_reward']]))) + '\n')


class ValidationPlotter(CoreCallback):
    def __init__(self):
        super().__init__()

    def on_episode_end(self, *args, **kwargs):
        if 'episode_number' in kwargs and 'action_number' in kwargs and 'episode_reward' in kwargs:
            episode_number = kwargs['episode_number']
            action_number = kwargs['action_number']
            episode_reward = kwargs['episode_reward']


class Visualizer(CoreCallback):
    def __init__(self, model, predicate=lambda x: True):
        super().__init__()

        self.model = model
        self.predicate = predicate

    def on_action_begin(self, *args, **kwargs):
        inputs = []
        if 'state' in kwargs:
            inputs.append(np.expand_dims(kwargs['state'], axis=0))

        if len(self.model.inputs) == 2 and 'action' in kwargs:
            inputs.append(np.expand_dims(kwargs['action'], axis=0))
