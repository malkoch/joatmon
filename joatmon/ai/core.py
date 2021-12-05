import random

import gym

from joatmon.core import CoreObject


class CoreBuffer(list, CoreObject):
    def __init__(self, values, batch_size):
        super(CoreBuffer, self).__init__()

        self.values = values
        self.batch_size = batch_size

    def __contains__(self, element):
        return element in self.values

    def __getitem__(self, idx):
        return self.values[idx]

    def __iter__(self):
        for value in self.values:
            yield value

    def __len__(self):
        return len(self.values)

    def add(self, element):
        self.values.append(element)

    def sample(self):
        return random.sample(self, self.batch_size)  # need to implement own random sampling algorithm


class CoreMemory(CoreObject):
    """
    Abstract base class for all implemented memory.

    Do not use this abstract base class directly but instead use one of the concrete memory implemented.

    To implement your own memory, you have to implement the following methods:

    - `remember`
    - `sample`
    """

    def __init__(self, buffer, batch_size):
        super(CoreMemory, self).__init__()

        if not isinstance(buffer, CoreBuffer):
            buffer = CoreBuffer([], batch_size)
        else:
            buffer.batch_size = batch_size

        self.buffer = buffer

    def __contains__(self, element):
        return element in self.buffer

    def __getitem__(self, idx):
        return self.buffer[idx]

    def __iter__(self):
        for value in self.buffer:
            yield value

    def __len__(self):
        return len(self.buffer)

    def remember(self, element):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.buffer.add(element)

    def sample(self):
        """
        Sample an experience replay batch with size.

        # Returns
            batch (abstract): Randomly selected batch
            from experience replay memory.
        """
        return self.buffer.sample()


class CoreCallback(CoreObject):
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
        pass

    def on_agent_end(self, *args, **kwargs):
        """
        Called at end of each agent play
        """
        pass

    def on_episode_begin(self, *args, **kwargs):
        """
        Called at beginning of each game episode
        """
        pass

    def on_episode_end(self, *args, **kwargs):
        """
        Called at end of each game episode
        """
        pass

    def on_action_begin(self, *args, **kwargs):
        """
        Called at beginning of each agent action
        """
        pass

    def on_action_end(self, *args, **kwargs):
        """
        Called at end of each agent action
        """
        pass

    def on_replay_begin(self, *args, **kwargs):
        """
        Called at beginning of each nn replay
        """
        pass

    def on_replay_end(self, *args, **kwargs):
        """
        Called at end of each nn replay
        """
        pass


class CoreEnv(CoreObject, gym.Env):
    """
    The abstract game class that is used by all agents. This class has the exact same API that OpenAI Gym uses so that integrating
    with it is trivial. In contrast to the OpenAI Gym implementation, this class only defines the abstract methods without any actual implementation.

    To implement your own game, you need to define the following methods:

    - `seed`
    - `reset`
    - `step`
    - `render`
    - `close`

    Refer to the [Gym documentation](https://gym.openai.com/docs/#environment).
    """

    def __init__(self):
        super(CoreEnv, self).__init__()

    def close(self):
        """
        Override in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        raise NotImplementedError

    def render(self, mode: str = 'human'):
        """
        Renders the game.

        The set of supported modes varies per game. (And some game do not support rendering at all.)

        # Arguments
            mode (str): The mode to render with. (default is 'human')
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the state of the game and returns an initial observation.

        # Returns
            observation (abstract): The initial observation of the space. Initial reward is assumed to be 0.
        """
        raise NotImplementedError

    def seed(self, seed=None):
        """
        set the seed
        """
        raise NotImplementedError

    def step(self, action):
        """
        Run one timestep of the game's dynamics.

        Accepts an action and returns a tuple (observation, reward, done, info).

        # Arguments
            action (abstract): An action provided by the game.

        # Returns
            observation (abstract): Agent's observation of the current game.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        raise NotImplementedError


class CorePolicy(CoreObject):
    """
    Abstract base class for all implemented policy.

    Do not use this abstract base class directly but
    instead use one of the concrete policy implemented.

    To implement your own policy, you have to implement the following methods:

    - `decay`
    - `use_network`
    """

    def __init__(self):
        super(CorePolicy, self).__init__()

    def reset(self):
        """
        reset
        """
        raise NotImplementedError

    def decay(self):
        """
        Decaying the epsilon / sigma value of the policy.
        """
        raise NotImplementedError

    def use_network(self):
        """
        Sample an experience replay batch with size.

        # Returns
            use (bool): Boolean value for using the nn.
        """
        raise NotImplementedError


class CoreRandom(CoreObject):
    """
    Abstract base class for all implemented random processes.

    Do not use this abstract base class directly but instead
    use one of the concrete random processes implemented.

    To implement your own random processes,
    you have to implement the following methods:

    - `decay`
    - `sample`
    - `reset`
    """

    def __init__(self):
        super(CoreRandom, self).__init__()

    def reset(self):
        """
        Reset random state.
        """
        raise NotImplementedError

    def decay(self):
        """
        decay
        """
        raise NotImplementedError

    def sample(self):
        """
        Sample random state.

        # Returns
            sample (abstract): Random state.
        """
        raise NotImplementedError


class CoreModel(CoreObject):
    """
    Abstract base class for all implemented nn.

    Do not use this abstract base class directly
    but instead use one of the concrete nn implemented.

    To implement your own nn, you have to implement the following methods:

    - `act`
    - `replay`
    - `load`
    - `save`
    """

    def __init__(self):
        super(CoreModel, self).__init__()

    def load(self):
        """
        load
        """
        raise NotImplementedError

    def save(self):
        """
        save
        """
        raise NotImplementedError

    def predict(self):
        """
        Get the action for given state.

        Accepts a state and returns an abstract action.

        # Arguments
            state (abstract): Current state of the game.

        # Returns
            action (abstract): Network's predicted action for given state.
        """
        raise NotImplementedError

    def train(self):
        """
        Train the nn with given batch.

        # Arguments
            batch (abstract): Mini Batch from Experience Replay Memory.
        """
        raise NotImplementedError

    def evaluate(self):
        """
        evaluate
        """
        raise NotImplementedError
