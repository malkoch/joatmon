import numpy as np


class CoreRandom(object):
    """
    Process a state.

    This method accepts a state and processes it for use in reinforcement learning. The state is resized to 84x84 and the color channels are moved to the second dimension.

    Args:
        state (numpy array): The input state as a numpy array.

    Returns:
        numpy array: The processed state as a numpy array.
    """

    def __init__(self):
        super(CoreRandom, self).__init__()

    def reset(self):
        """
        Reset the random state.

        This method should be overridden by any subclass.
        """
        raise NotImplementedError

    def decay(self):
        """
        Decay the random state.

        This method should be overridden by any subclass.
        """
        raise NotImplementedError

    def sample(self):
        """
        Sample from the random state.

        This method should be overridden by any subclass.

        Returns:
            sample (abstract): The sampled random state.
        """
        raise NotImplementedError


# another class to make the noise decayed, parent class for all
class GaussianRandom(CoreRandom):
    """
    Gaussian Noise

    This class generates a Gaussian noise process.

    Attributes:
        mu (float): The mean of the Gaussian distribution.
        size (int): The size of the output sample.
        sigma (float): The standard deviation of the Gaussian distribution.
        sigma_min (float): The minimum standard deviation.
        decay_steps (int): The number of steps over which the standard deviation decays from sigma to sigma_min.

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
        """
        Reset the random state.

        This method resets the number of steps and the current standard deviation.
        """
        self.n_steps = 0
        self.current_sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)

    def decay(self):
        """
        Decay the random state.

        This method increments the number of steps and updates the current standard deviation.
        """
        self.n_steps += 1
        self.current_sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)

    def sample(self):
        """
        Sample from the random state.

        This method generates a sample from a Gaussian distribution with the current mean and standard deviation.

        Returns:
            x (numpy array): The sampled random state.
        """
        x = np.random.normal(loc=self.mu, scale=self.current_sigma, size=self.size)
        return x


class OrnsteinUhlenbeck(CoreRandom):
    """
    Ornstein Uhlenbeck Process

    This class generates an Ornstein Uhlenbeck noise process.

    Attributes:
        dt (float): The time increment.
        mu (float): The mean of the distribution.
        size (int): The size of the output sample.
        sigma (float): The standard deviation of the distribution.
        theta (float): The rate of mean reversion.
        sigma_min (float): The minimum standard deviation.
        decay_steps (int): The number of steps over which the standard deviation decays from sigma to sigma_min.

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
        """
        Reset the random state.

        This method resets the number of steps, the current standard deviation, and the previous state.
        """
        self.n_steps = 0
        self.x_prev = np.ones(self.size) * self.mu
        self.current_sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)

    def decay(self):
        """
        Decay the random state.

        This method increments the number of steps and updates the current standard deviation.
        """
        self.n_steps += 1
        self.current_sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)

    def sample(self):
        """
        Sample from the random state.

        This method generates a sample from an Ornstein Uhlenbeck process with the current parameters.

        Returns:
            x (numpy array): The sampled random state.
        """
        x = (
                self.x_prev
                + self.theta * (self.mu - self.x_prev) * self.dt
                + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        )
        self.x_prev = x
        return x
