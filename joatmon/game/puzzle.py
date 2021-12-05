import random

import cv2
import numpy as np

from joatmon.ai.core import CoreEnv

__all__ = ['Puzzle2048']


class Puzzle2048(CoreEnv):
    def __init__(self, size):
        super().__init__()

        self.size = size
        self.matrix = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.reset()

    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass

    def __repr__(self):
        return str(self)

    def __str__(self):
        ret = '-' * 33 + '\n'
        for row in self.matrix:
            ret += '|'
            for val in row:
                if val != 0:
                    v = 2 ** val
                    ret += '{:7d}'.format(v)
                else:
                    ret += ' ' * 7
                ret += '|'
            ret += '\n'
            ret += '-' * 33 + '\n'
        # ret += '-' * 33
        return ret

    def _check(self):
        for row in range(self.size):
            for col in range(self.size):
                val = self.matrix[row][col]
                if val == 0:
                    return True

                if row == 0:
                    if col == 0:
                        vals = [self.matrix[row + 1][col], self.matrix[row][col + 1]]
                    elif col == self.size - 1:
                        vals = [self.matrix[row + 1][col], self.matrix[row][col - 1]]
                    else:
                        vals = [self.matrix[row + 1][col], self.matrix[row][col - 1], self.matrix[row][col + 1]]
                elif row == self.size - 1:
                    if col == 0:
                        vals = [self.matrix[row - 1][col], self.matrix[row][col + 1]]
                    elif col == self.size - 1:
                        vals = [self.matrix[row - 1][col], self.matrix[row][col - 1]]
                    else:
                        vals = [self.matrix[row - 1][col], self.matrix[row][col - 1], self.matrix[row][col + 1]]
                else:
                    if col == 0:
                        vals = [self.matrix[row - 1][col], self.matrix[row + 1][col], self.matrix[row][col + 1]]
                    elif col == self.size - 1:
                        vals = [self.matrix[row - 1][col], self.matrix[row + 1][col], self.matrix[row][col - 1]]
                    else:
                        vals = [self.matrix[row - 1][col], self.matrix[row + 1][col], self.matrix[row][col - 1], self.matrix[row][col + 1]]
                if val in vals:
                    return True
        return False

    def _fill(self, action):
        ret = False
        if action == 0:
            for row in range(self.size):
                for col in range(self.size - 1, 0, -1):
                    if self.matrix[row][col] == 0 and self.matrix[row][col - 1] != 0:
                        self.matrix[row][col] = self.matrix[row][col - 1]
                        self.matrix[row][col - 1] = 0
                        ret = True
        elif action == 1:
            for col in range(self.size):
                for row in range(self.size - 1, 0, -1):
                    if self.matrix[row][col] == 0 and self.matrix[row - 1][col] != 0:
                        self.matrix[row][col] = self.matrix[row - 1][col]
                        self.matrix[row - 1][col] = 0
                        ret = True
        elif action == 2:
            for row in range(self.size):
                for col in range(self.size - 1):
                    if self.matrix[row][col] == 0 and self.matrix[row][col + 1] != 0:
                        self.matrix[row][col] = self.matrix[row][col + 1]
                        self.matrix[row][col + 1] = 0
                        ret = True
        elif action == 3:
            for col in range(self.size):
                for row in range(self.size - 1):
                    if self.matrix[row][col] == 0 and self.matrix[row + 1][col] != 0:
                        self.matrix[row][col] = self.matrix[row + 1][col]
                        self.matrix[row + 1][col] = 0
                        ret = True
        else:
            pass
        return ret

    def _merge(self, action):
        reward = 9e-1
        if action == 0:
            for row in range(self.size):
                for col in range(self.size - 1, 0, -1):
                    val1 = self.matrix[row][col]
                    val2 = self.matrix[row][col - 1]
                    if val1 == val2 and val1 != 0 and val2 != 0:
                        self.matrix[row][col] += 1
                        self.matrix[row][col - 1] = 0
                        reward += 2 ** val1
        elif action == 1:
            for col in range(self.size):
                for row in range(self.size - 1, 0, -1):
                    val1 = self.matrix[row][col]
                    val2 = self.matrix[row - 1][col]
                    if val1 == val2 and val1 != 0 and val2 != 0:
                        self.matrix[row][col] += 1
                        self.matrix[row - 1][col] = 0
                        reward += 2 ** val1
        elif action == 2:
            for row in range(self.size):
                for col in range(self.size - 1):
                    val1 = self.matrix[row][col]
                    val2 = self.matrix[row][col + 1]
                    if val1 == val2 and val1 != 0 and val2 != 0:
                        self.matrix[row][col] += 1
                        self.matrix[row][col + 1] = 0
                        reward += 2 ** val1
        elif action == 3:
            for col in range(self.size):
                for row in range(self.size - 1):
                    val1 = self.matrix[row][col]
                    val2 = self.matrix[row + 1][col]
                    if val1 == val2 and val1 != 0 and val2 != 0:
                        self.matrix[row][col] += 1
                        self.matrix[row + 1][col] = 0
                        reward += 2 ** val1
        else:
            pass
        return reward

    def _randomize(self):
        indexes = []
        for row in range(self.size):
            for col in range(self.size):
                if self.matrix[row][col] == 0:
                    indexes.append((row, col))
        if indexes:
            row, col = random.sample(indexes, 1)[0]
            self.matrix[row][col] = np.random.choice([1, 2], p=[0.9, 0.1])

    def close(self):
        pass

    def render(self, mode='human'):
        # print(self)
        rect_size = 75
        state_size = rect_size * self.size
        state = np.zeros((state_size, state_size, 3)).astype('uint8')

        for row in range(self.size):
            for col in range(self.size):
                cv2.rectangle(state, (col * rect_size, row * rect_size), ((col + 1) * rect_size, (row + 1) * rect_size), (255, 255, 255), 1)
                if self.matrix[row][col] != 0:
                    w, h = cv2.getTextSize(str(2 ** self.matrix[row][col]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.putText(state, str(2 ** self.matrix[row][col]), (int((col + 1) * rect_size - w[0]), int((row + 0.5) * rect_size + h)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('state', state)
        cv2.waitKey(1)

    def reset(self):
        self.matrix = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self._randomize()
        self._randomize()

        return self.matrix

    def seed(self, seed=None):
        pass

    def step(self, action):
        # 0 -> right
        # 1 -> down
        # 2 -> left
        # 3 -> up
        randomize = False
        # control fill, merge and randomize sometimes they do not work
        for _ in range(self.size):
            randomize |= self._fill(action)
        reward = self._merge(action)
        for _ in range(self.size):
            randomize |= self._fill(action)
        if randomize:
            self._randomize()
        return self.matrix, np.log(reward), not self._check(), {}
