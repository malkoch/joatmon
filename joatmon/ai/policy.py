import numpy as np

from joatmon.ai.core import CorePolicy


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
