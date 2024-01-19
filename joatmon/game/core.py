import gym


class CoreEnv(gym.Env):
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
        """
        Initialize a new CoreEnv instance.
        """
        super(CoreEnv, self).__init__()

    def close(self):
        """
        Clean up the environment's resources.

        Raises:
            NotImplementedError: This method needs to be implemented in the subclasses.
        """
        raise NotImplementedError

    def render(self, mode: str = 'human'):
        """
        Render the environment.

        Args:
            mode (str, optional): The mode to use for rendering. Defaults to 'human'.

        Raises:
            NotImplementedError: This method needs to be implemented in the subclasses.
        """
        raise NotImplementedError

    def reset(self, *args):
        """
        Reset the environment to its initial state and return the initial observation.

        Args:
            *args: Variable length argument list.

        Raises:
            NotImplementedError: This method needs to be implemented in the subclasses.
        """
        raise NotImplementedError

    def seed(self, seed=None):
        """
        Set the seed for the environment's random number generator.

        Args:
            seed (int, optional): The seed to use. Defaults to None.

        Raises:
            NotImplementedError: This method needs to be implemented in the subclasses.
        """
        raise NotImplementedError

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.

        Args:
            action: An action to take in the environment.

        Returns:
            observation: The agent's observation of the current environment.
            reward (float): The amount of reward returned after previous action.
            done (bool): Whether the episode has ended.
            info (dict): Contains auxiliary diagnostic information.

        Raises:
            NotImplementedError: This method needs to be implemented in the subclasses.
        """
        raise NotImplementedError

    def goal(self):
        """
        Get the goal state of the environment.

        Raises:
            NotImplementedError: This method needs to be implemented in the subclasses.
        """
        raise NotImplementedError
