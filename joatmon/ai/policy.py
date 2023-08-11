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
        pass

    def decay(self):
        pass

    def use_network(self):
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
        pass

    def decay(self):
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_value)

    def use_network(self):
        return np.random.uniform() >= self.epsilon
