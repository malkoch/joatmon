import os

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')


class CoreCallback(object):
    """
    Abstract base class for all implemented callback.

    Do not use this abstract base class directly but instead use one of the concrete callback implemented.

    To implement your own callback, you have to implement the following methods:

    - `on_action_begin`
    - `on_action_end`
    - `on_replay_begin`
    - `on_replay_end`
    - `on_episode_begin`
    - `on_episode_end`
    - `on_agent_begin`
    - `on_agent_end`
    """

    def __init__(self):
        super(CoreCallback, self).__init__()

    def on_agent_begin(self, *args, **kwargs):
        """
        Called at beginning of each agent play
        """

    def on_agent_end(self, *args, **kwargs):
        """
        Called at end of each agent play
        """

    def on_episode_begin(self, *args, **kwargs):
        """
        Called at beginning of each game episode
        """

    def on_episode_end(self, *args, **kwargs):
        """
        Called at end of each game episode
        """

    def on_action_begin(self, *args, **kwargs):
        """
        Called at beginning of each agent action
        """

    def on_action_end(self, *args, **kwargs):
        """
        Called at end of each agent action
        """

    def on_replay_begin(self, *args, **kwargs):
        """
        Called at beginning of each nn replay
        """

    def on_replay_end(self, *args, **kwargs):
        """
        Called at end of each nn replay
        """


class CallbackList(CoreCallback):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, callbacks):
        super(CallbackList, self).__init__()

        self.callbacks = callbacks

    def on_action_begin(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for callback in self.callbacks:
            callback.on_action_begin(*args, **kwargs)

    def on_action_end(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for callback in self.callbacks:
            callback.on_action_end(*args, **kwargs)

    def on_agent_begin(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for callback in self.callbacks:
            callback.on_agent_begin(*args, **kwargs)

    def on_agent_end(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for callback in self.callbacks:
            callback.on_agent_end(*args, **kwargs)

    def on_episode_begin(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for callback in self.callbacks:
            callback.on_episode_begin(*args, **kwargs)

    def on_episode_end(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for callback in self.callbacks:
            callback.on_episode_end(*args, **kwargs)

    def on_replay_begin(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for callback in self.callbacks:
            callback.on_replay_begin(*args, **kwargs)

    def on_replay_end(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for callback in self.callbacks:
            callback.on_replay_end(*args, **kwargs)


class Loader(CoreCallback):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, model, run_path, interval):
        super().__init__()

        self.network = model
        self.run_path = run_path
        self.interval = interval

    def on_agent_begin(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        # weights path should be run path
        self.network.load(self.run_path)
        self.network.save(self.run_path)

    def on_agent_end(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.network.save(self.run_path)

    def on_episode_end(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if 'episode_number' in kwargs:
            if kwargs['episode_number'] % self.interval == 0:
                self.network.save(self.run_path)


class Renderer(CoreCallback):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, environment):
        super().__init__()

        self.environment = environment

    def on_action_end(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.environment.render()

    def on_episode_begin(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.environment.render()


class TrainLogger(CoreCallback):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

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
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if 'agent_headers' in kwargs:
            with open(self.agent_data_path, 'w') as file:
                file.write(','.join(kwargs['agent_headers']) + '\n')

        if 'network_headers' in kwargs:
            with open(self.network_data_path, 'w') as file:
                file.write(','.join(kwargs['network_headers']) + '\n')

    def on_episode_begin(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """

    def on_episode_end(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if 'episode_number' in kwargs and 'action_number' in kwargs and 'episode_reward' in kwargs:
            end = '\n' if kwargs['episode_number'] % self.interval == 0 else ''
            print(
                self.episode_end_message_raw.format(
                    kwargs['episode_number'], kwargs['action_number'], kwargs['episode_reward']
                ),
                end=end,
            )
            with open(self.agent_data_path, 'a') as file:
                file.write(
                    ','.join(
                        list(map(str, [kwargs['episode_number'], kwargs['action_number'], kwargs['episode_reward']]))
                    )
                    + '\n'
                )

    def on_replay_end(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if 'loss' in kwargs:
            if not isinstance(kwargs['loss'], (list, tuple)):
                kwargs['loss'] = [kwargs['loss']]

            with open(self.network_data_path, 'a') as file:
                file.write(','.join(list(map(str, kwargs['loss']))) + '\n')


class ValidationLogger(CoreCallback):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, run_path, interval):
        super().__init__()

        self.run_path = run_path
        self.interval = interval

        if not os.path.exists(run_path):
            os.makedirs(run_path)

        self.agent_data_path = self.run_path + 'test-agent-data.csv'
        self.episode_end_message_raw = '\repisode {:05d} ended in {:04d} actions, total reward: {:+08.2f}'

    def on_agent_begin(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        with open(self.agent_data_path, 'w') as file:
            file.write('episode_number,action_number,episode_reward\n')

    def on_episode_begin(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """

    def on_episode_end(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if 'episode_number' in kwargs and 'action_number' in kwargs and 'episode_reward' in kwargs:
            end = '\n' if kwargs['episode_number'] % self.interval == 0 else ''
            print(
                self.episode_end_message_raw.format(
                    kwargs['episode_number'], kwargs['action_number'], kwargs['episode_reward']
                ),
                end=end,
            )
            with open(self.agent_data_path, 'a') as file:
                file.write(
                    ','.join(
                        list(map(str, [kwargs['episode_number'], kwargs['action_number'], kwargs['episode_reward']]))
                    )
                    + '\n'
                )


class Visualizer(CoreCallback):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, model, predicate=lambda x: True):
        super().__init__()

        self.model = model
        self.predicate = predicate

    def on_action_begin(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        inputs = []
        if 'state' in kwargs:
            inputs.append(np.expand_dims(kwargs['state'], axis=0))

        if len(self.model.inputs) == 2 and 'action' in kwargs:
            inputs.append(np.expand_dims(kwargs['action'], axis=0))


class LivePlotter(CoreCallback):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    @staticmethod
    def live_plotter(y_num, size):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
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
            plt.pause(0.1)

        return f
