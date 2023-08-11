import numpy as np


class CoreRandom(object):
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


# another class to make the noise decayed, parent class for all
class GaussianRandom(CoreRandom):
    """
    Gaussian Noise

    # Arguments
        mu (float): .
        size (int): .
        sigma (float): .
        sigma_min (float): .
        decay_steps (int): .
    """

    def __init__(self, mu=0.0, size=2, sigma=0.1, sigma_min=0.01, decay_steps=200000):
        super().__init__()

        self.mu = mu
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.size = size

        self.m = -float(sigma - sigma_min) / float(decay_steps)
        self.c = sigma

        self.n_steps = 0
        self.current_sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        self.reset()

    def reset(self):
        self.n_steps = 0
        self.current_sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)

    def decay(self):
        self.n_steps += 1
        self.current_sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)

    def sample(self):
        x = np.random.normal(loc=self.mu, scale=self.current_sigma, size=self.size)
        return x


class OrnsteinUhlenbeck(CoreRandom):
    """
    Ornstein Uhlenbeck Process

    # Arguments
        dt (float): .
        mu (float): .
        size (int): .
        sigma (float): .
        theta (float): .
        sigma_min (float): .
        decay_steps (int): .
    """

    def __init__(self, dt=1.0, mu=0.0, size=2, sigma=0.1, theta=0.15, sigma_min=0.01, decay_steps=200000):
        super().__init__()

        self.dt = dt
        self.mu = mu
        self.size = size
        self.sigma = sigma
        self.theta = theta
        self.sigma_min = sigma_min

        self.m = -float(sigma - sigma_min) / float(decay_steps)
        self.c = sigma

        self.n_steps = 0
        self.x_prev = np.ones(self.size) * self.mu
        self.current_sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        self.reset()

    def reset(self):
        self.n_steps = 0
        self.x_prev = np.ones(self.size) * self.mu
        self.current_sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)

    def decay(self):
        self.n_steps += 1
        self.current_sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        return x
