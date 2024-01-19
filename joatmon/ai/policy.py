import numpy as np


class CorePolicy(object):
    """
    Abstract base class for all implemented policies.

    This class should not be used directly. Instead, use one of the concrete policies implemented.
    To implement your own policy, you have to implement the following methods: `decay`, `reset`, `use_network`.
    """

    def __init__(self):
        super(CorePolicy, self).__init__()

    def reset(self):
        """
        Reset the policy.

        This method should be overridden by any subclass.
        """
        raise NotImplementedError

    def decay(self):
        """
        Decay the policy.

        This method should be overridden by any subclass.
        """
        raise NotImplementedError

    def use_network(self):
        """
        Determine whether to use the network for decision making.

        This method should be overridden by any subclass.

        Returns:
            use (bool): Boolean value for using the network.
        """
        raise NotImplementedError


class GreedyQPolicy(CorePolicy):
    """
    Greedy Q Policy

    This class implements a policy that always selects the action with the highest expected reward.
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        """
        Reset the policy.

        This method is currently a placeholder and does nothing.
        """

    def decay(self):
        """
        Decay the policy.

        This method is currently a placeholder and does nothing.
        """

    def use_network(self):
        """
        Determine whether to use the network for decision making.

        For a GreedyQPolicy, this always returns True.

        Returns:
            use (bool): Boolean value for using the network.
        """
        return True


class EpsilonGreedyPolicy(CorePolicy):
    """
    Epsilon Greedy Policy

    This class implements a policy that selects a random action with probability epsilon and the action with the highest expected reward with probability 1 - epsilon.

    Attributes:
        epsilon (float): The probability of selecting a random action.
        min_value (float): The minimum value that epsilon can decay to.
        epsilon_decay (float): The amount by which epsilon is reduced at each step.

    Args:
        max_value (float): The initial value of epsilon.
        min_value (float): The minimum value that epsilon can decay to.
        decay_steps (int): The number of steps over which epsilon decays from max_value to min_value.
    """

    def __init__(self, max_value=1.0, min_value=0.0, decay_steps=1):
        super().__init__()

        self.epsilon = max_value
        self.min_value = min_value
        self.epsilon_decay = (max_value - min_value) / decay_steps

    def reset(self):
        """
        Reset the policy.

        This method is currently a placeholder and does nothing.
        """

    def decay(self):
        """
        Decay the policy.

        This method reduces the value of epsilon by epsilon_decay, down to a minimum of min_value.
        """
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_value)

    def use_network(self):
        """
        Determine whether to use the network for decision making.

        For an EpsilonGreedyPolicy, this returns True with probability 1 - epsilon and False with probability epsilon.

        Returns:
            use (bool): Boolean value for using the network.
        """
        return np.random.uniform() >= self.epsilon
