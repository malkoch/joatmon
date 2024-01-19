import random

__all__ = ['TilesEnv']

from joatmon.game.core import CoreEnv


class TilesEnv(CoreEnv):
    """
    TilesEnv class for creating a 2048 game environment.

    This class inherits from CoreEnv and provides methods for game operations such as resetting the game, rendering the game state, and performing a step in the game.

    Attributes:
        size (int): The size of the game grid.
        tiles (list): The game grid represented as a 2D list.
    """

    def __init__(self, size):
        """
        Initialize a new TilesEnv instance.

        Args:
            size (int): The size of the game grid.
        """
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
        Find the position of the empty tile in the game grid.

        Returns:
            tuple: The position of the empty tile.
        """
        for y in range(self.size):
            for x in range(self.size):
                if self.tiles[y][x] == 0:
                    return y, x
        return None, None

    def _swap(self, pos1, pos2):
        """
        Swap the tiles at the specified positions.

        Args:
            pos1 (tuple): The position of the first tile.
            pos2 (tuple): The position of the second tile.
        """
        temp = self.tiles[pos2[0]][pos2[1]]
        self.tiles[pos2[0]][pos2[1]] = self.tiles[pos1[0]][pos1[1]]
        self.tiles[pos1[0]][pos1[1]] = temp

    def close(self):
        """
        Close the game environment.
        """

    def render(self, mode='human'):
        """
        Render the game state.

        Args:
            mode (str, optional): The mode to use for rendering. Defaults to 'human'.
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
        Reset the game state.

        Returns:
            list: The reset game grid.
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
        Set the seed for the game's random number generator.

        Args:
            seed (int, optional): The seed to use. Defaults to None.
        """

    def step(self, action):
        """
        Perform a step in the game.

        Args:
            action (int): The action to perform.

        Returns:
            tuple: The new game state, the reward obtained, whether the game is over, and additional info.
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
