import gym


class CoreSpace(gym.Space):
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

    def __init__(self, shape=None, dtype=None):
        super(CoreSpace, self).__init__(shape, dtype)

    @property
    def is_np_flattenable(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        raise NotImplementedError

    def sample(self, mask=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        raise NotImplementedError

    def contains(self, x) -> bool:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        raise NotImplementedError


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

    reward_range = (-float('inf'), float('inf'))
    action_space: CoreSpace
    observation_space: CoreSpace

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

    def reset(self, *args):
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

    def goal(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        raise NotImplementedError
