import numpy as np


class CorePolicy(object):
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


class GreedyQPolicy(CorePolicy):
    def __init__(self):
        super().__init__()

    def reset(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """

    def decay(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """

    def use_network(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return True


class EpsilonGreedyPolicy(CorePolicy):
    """
    Epsilon Greedy

    # Arguments
        max_value (float): .
        min_value (float): .
        decay_steps (int): .
    """

    def __init__(self, max_value=1.0, min_value=0.0, decay_steps=1):
        super().__init__()

        self.epsilon = max_value
        self.min_value = min_value
        self.epsilon_decay = (max_value - min_value) / decay_steps

    def reset(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """

    def decay(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_value)

    def use_network(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return np.random.uniform() >= self.epsilon
