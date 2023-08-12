import random

__all__ = ['TilesEnv']

from joatmon.game.core import CoreEnv


class TilesEnv(CoreEnv):
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

    def __init__(self, size):
        super().__init__()

        self.size = size

        self.tiles = []
        for y in range(self.size):
            row = []
            for x in range(self.size):
                if y * self.size + x + 1 == self.size ** 2:
                    row.append(0)
                else:
                    row.append(y * self.size + x + 1)
            self.tiles.append(row)

    def _find_empty(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for y in range(self.size):
            for x in range(self.size):
                if self.tiles[y][x] == 0:
                    return y, x
        return None, None

    def _swap(self, pos1, pos2):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        temp = self.tiles[pos2[0]][pos2[1]]
        self.tiles[pos2[0]][pos2[1]] = self.tiles[pos1[0]][pos1[1]]
        self.tiles[pos1[0]][pos1[1]] = temp

    def close(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """

    def render(self, mode='human'):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        to_print = ''
        for y in range(self.size):
            line = ''
            for x in range(self.size):
                line += '{:2d} '.format(self.tiles[y][x]) if self.tiles[y][x] != 0 else '   '
            to_print += line + '\n'
        print(to_print)

    def reset(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        r = [i + 1 for i in range(self.size ** 2)]
        random.shuffle(r)

        for i, elem in enumerate(r):
            div, mod = divmod(i, self.size)
            if elem == self.size ** 2:
                self.tiles[div][mod] = 0
            else:
                self.tiles[div][mod] = elem

    def seed(self, seed=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """

    def step(self, action):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        empty_y, empty_x = self._find_empty()

        if empty_y is not None and empty_x is not None:
            if isinstance(action, int):
                if 0 <= action < 4:
                    if action == 0:
                        if empty_x + 1 < self.size:
                            self._swap([empty_y, empty_x], [empty_y, empty_x + 1])
                    elif action == 1:
                        if empty_y - 1 >= 0:
                            self._swap([empty_y, empty_x], [empty_y - 1, empty_x])
                    elif action == 2:
                        if empty_x - 1 >= 0:
                            self._swap([empty_y, empty_x], [empty_y, empty_x - 1])
                    elif action == 3:
                        if empty_y + 1 < self.size:
                            self._swap([empty_y, empty_x], [empty_y + 1, empty_x])
