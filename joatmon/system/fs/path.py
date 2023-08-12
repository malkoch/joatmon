import os

__all__ = ['Path']


class Path:
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

    def __init__(self, path=''):
        if path is None or path == '':
            path = os.getcwd()
        self.path = path

    def __str__(self):
        return self.path

    def __repr__(self):
        return str(self)

    def __truediv__(self, other):
        if isinstance(other, Path):
            other = other.path

        if other is None or other == '' or other == '/':
            other = '.'

        if other == '.':
            return self

        return Path(os.path.join(self.path, other))

    def __itruediv__(self, other):
        new = self / other
        self.path = new.path

        return self

    def __abs__(self):
        if not self.path.startswith('/'):
            cwd = os.getcwd()
            new = Path(cwd) / self

            return new

        return self

    def exists(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return os.path.exists(self.path)

    def isdir(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return os.path.isdir(self.path)

    def isfile(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return os.path.isfile(self.path)

    def mkdir(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return os.mkdir(self.path)

    def list(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return os.listdir(self.path)
