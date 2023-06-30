import cv2
import numpy as np
import pygame

__all__ = ['CubeEnv']

from joatmon.game.core import CoreEnv

cube_colors = ['blue', 'red', 'yellow', 'green', 'orange', 'white']
cube_colors = list(map(lambda x: pygame.color.THECOLORS[x], cube_colors))

screen_size = 400


def flip(pos):
    return pos[0], -pos[1] + screen_size


def project_points(points, q, view, vertical=(0, 1, 0)):
    points = np.asarray(points)
    view = np.asarray(view)

    xdir = np.cross(vertical, view).astype(float)

    if np.all(xdir == 0):
        raise ValueError("vertical is parallel to v")

    xdir /= np.sqrt(np.dot(xdir, xdir))

    ydir = np.cross(view, xdir)
    ydir /= np.sqrt(np.dot(ydir, ydir))

    v2 = np.dot(view, view)
    zdir = view / np.sqrt(v2)

    r = q.as_rotation_matrix()
    rpts = np.dot(points, r.T)

    dpoint = rpts - view
    dpoint_view = np.dot(dpoint, view).reshape(dpoint.shape[:-1] + (1,))
    dproj = -dpoint * v2 / dpoint_view

    trans = list(range(1, dproj.ndim)) + [0]
    return np.array(
        [
            np.dot(dproj, xdir), np.dot(dproj, ydir), -np.dot(dpoint, zdir)
        ]
    ).transpose(trans)


def resize(pos):
    return pos * (screen_size // 4) + (screen_size // 2)


class Quaternion:
    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)

    def __mul__(self, other):
        sxr = self.x.reshape(self.x.shape[:-1] + (4, 1))
        oxr = other.x.reshape(other.x.shape[:-1] + (1, 4))

        prod = sxr * oxr
        return_shape = prod.shape[:-1]
        prod = prod.reshape((-1, 4, 4)).transpose((1, 2, 0))

        ret = np.array(
            [
                (prod[0, 0] - prod[1, 1] - prod[2, 2] - prod[3, 3]),
                (prod[0, 1] + prod[1, 0] + prod[2, 3] - prod[3, 2]),
                (prod[0, 2] - prod[1, 3] + prod[2, 0] + prod[3, 1]),
                (prod[0, 3] + prod[1, 2] - prod[2, 1] + prod[3, 0])
            ], dtype=np.float, order='F'
        ).T
        return self.__class__(ret.reshape(return_shape))

    def __repr__(self):
        return "Quaternion:\n" + self.x.__repr__()

    def as_rotation_matrix(self):
        v, theta = self.as_v_theta()

        shape = theta.shape
        theta = theta.reshape(-1)
        v = v.reshape(-1, 3).T
        c = np.cos(theta)
        s = np.sin(theta)

        mat = np.array(
            [
                [v[0] * v[0] * (1 - c) + c, v[0] * v[1] * (1 - c) - v[2] * s, v[0] * v[2] * (1 - c) + v[1] * s],
                [v[1] * v[0] * (1 - c) + v[2] * s, v[1] * v[1] * (1 - c) + c, v[1] * v[2] * (1 - c) - v[0] * s],
                [v[2] * v[0] * (1 - c) - v[1] * s, v[2] * v[1] * (1 - c) + v[0] * s, v[2] * v[2] * (1 - c) + c]
            ], order='F'
        ).T
        return mat.reshape(shape + (3, 3))

    def as_v_theta(self):
        x = self.x.reshape((-1, 4)).T

        norm = np.sqrt((x ** 2).sum(0))
        theta = 2 * np.arccos(x[0] / norm)

        v = np.array(x[1:], order='F')
        v /= np.sqrt(np.sum(v ** 2, 0))

        v = v.T.reshape(self.x.shape[:-1] + (3,))
        theta = theta.reshape(self.x.shape[:-1])

        return v, theta

    @classmethod
    def from_v_theta(cls, v, theta):
        theta = np.asarray(theta)
        v = np.asarray(v)
        s = np.sin(0.5 * theta)
        c = np.cos(0.5 * theta)

        v = v * s / np.sqrt(np.sum(v * v, -1))
        x_shape = v.shape[:-1] + (4,)

        x = np.ones(x_shape).reshape(-1, 4)
        x[:, 0] = c.ravel()
        x[:, 1:] = v.reshape(-1, 3)
        x = x.reshape(x_shape)

        return cls(x)

    def rotate(self, points):
        m = self.as_rotation_matrix()
        return np.dot(points, m.T)


class Sticker:
    def __init__(self, color):
        self.color = color

    def draw(self, screen, points, view, rotation, vertical):
        points = project_points(points, rotation, view, vertical)[:, :2]
        points = list(map(resize, points))
        points = list(map(flip, points))
        pygame.draw.polygon(screen, cube_colors[self.color], points)
        pygame.draw.polygon(screen, pygame.color.THECOLORS['black'], points, 1)


class Face:
    def __init__(self, n, color, top_left, increment):
        self.n = n
        self.top_left = top_left
        self.increment = increment

        self.stickers = [[Sticker(color) for _ in range(n)] for _ in range(n)]

    def draw(self, screen, view, rotation, vertical):
        for i, row in enumerate(self.stickers):
            for j, sticker in enumerate(row):
                sticker.draw(
                    screen, (self.top_left + self.increment[0] * (i + 0) + self.increment[1] * (j + 0),
                             self.top_left + self.increment[0] * (i + 0) + self.increment[1] * (j + 1),
                             self.top_left + self.increment[0] * (i + 1) + self.increment[1] * (j + 1),
                             self.top_left + self.increment[0] * (i + 1) + self.increment[1] * (j + 0),
                             self.top_left + self.increment[0] * (i + 0) + self.increment[1] * (j + 0)), view, rotation, vertical
                )

    def rotate(self, times):
        for _ in range(times):
            self.rot90()

    def rotate_layer(self, layer):
        for ind in range(layer, self.n - layer - 1):
            tmp = self.stickers[layer][ind].color
            self.stickers[layer][ind].color = self.stickers[self.n - layer - ind - 1][layer].color
            self.stickers[self.n - layer - ind - 1][layer].color = self.stickers[self.n - layer - 1][self.n - layer - ind - 1].color
            self.stickers[self.n - layer - 1][self.n - layer - ind - 1].color = self.stickers[layer + ind][self.n - layer - 1].color
            self.stickers[layer + ind][self.n - layer - 1].color = tmp

    def rot90(self):
        for layer in range(self.n // 2):
            self.rotate_layer(layer)


class Cube:
    # blue left, red front, yellow top
    # green right, orange back, white bottom
    def __init__(self, n=3):
        self.n = n
        self.faces = {
            'L': Face(
                n, 0, np.asarray([-1, +1, -1]).astype('float32'),
                (np.asarray([0, -1, 0]).astype('float32') * (2.0 / n), np.asarray([0, 0, +1]).astype('float32') * (2.0 / n))
            ),
            'F': Face(
                n, 1, np.asarray([-1, +1, +1]).astype('float32'),
                (np.asarray([0, -1, 0]).astype('float32') * (2.0 / n), np.asarray([+1, 0, 0]).astype('float32') * (2.0 / n))
            ),
            'U': Face(
                n, 2, np.asarray([-1, +1, -1]).astype('float32'),
                (np.asarray([0, 0, +1]).astype('float32') * (2.0 / n), np.asarray([+1, 0, 0]).astype('float32') * (2.0 / n))
            ),
            'R': Face(
                n, 3, np.asarray([+1, +1, +1]).astype('float32'),
                (np.asarray([0, -1, 0]).astype('float32') * (2.0 / n), np.asarray([0, 0, -1]).astype('float32') * (2.0 / n))
            ),
            'B': Face(
                n, 4, np.asarray([+1, +1, -1]).astype('float32'),
                (np.asarray([0, -1, 0]).astype('float32') * (2.0 / n), np.asarray([-1, 0, 0]).astype('float32') * (2.0 / n))
            ),
            'D': Face(
                n, 5, np.asarray([-1, -1, +1]).astype('float32'),
                (np.asarray([0, 0, -1]).astype('float32') * (2.0 / n), np.asarray([+1, 0, 0]).astype('float32') * (2.0 / n))
            )
        }
        self.order = ['D', 'B', 'L', 'R', 'F', 'U']

        self.front = pygame.Surface((screen_size, screen_size))
        self.back = pygame.Surface((screen_size, screen_size))

        self.screen = pygame.Surface((screen_size, screen_size))

        self.view = (0, 0, 15)
        self.rotation = Quaternion.from_v_theta((1, -1, 0), -np.pi / 6)
        self.vertical = [0, 1, 0]

    def draw(self):
        self.back.fill(pygame.color.THECOLORS['black'])
        for face_name in self.order[:3]:
            self.faces[face_name].draw(self.back, list(map(lambda x: -x, self.view)), self.rotation, list(map(lambda x: -x, self.vertical)))
        cv2.imshow('back', cv2.cvtColor(np.swapaxes(pygame.surfarray.array3d(self.back), 0, 1), cv2.COLOR_RGB2BGR))

        self.front.fill(pygame.color.THECOLORS['black'])
        for face_name in self.order[3:]:
            self.faces[face_name].draw(self.front, self.view, self.rotation, self.vertical)
        cv2.imshow('front', cv2.cvtColor(np.swapaxes(pygame.surfarray.array3d(self.front), 0, 1), cv2.COLOR_RGB2BGR))

        cv2.waitKey(0)

    def swap_faces(self, faces):
        for row in range(self.n):
            for col in range(self.n):
                tmp = self.faces[faces[0]].stickers[row][col].color
                self.faces[faces[0]].stickers[row][col].color = self.faces[faces[1]].stickers[row][col].color
                self.faces[faces[1]].stickers[row][col].color = self.faces[faces[2]].stickers[row][col].color
                self.faces[faces[2]].stickers[row][col].color = self.faces[faces[3]].stickers[row][col].color
                self.faces[faces[3]].stickers[row][col].color = tmp

    def swap_layers(self, faces, layer):
        pass

    def u(self):
        self.faces['U'].rotate(1)
        for ind in range(self.n):
            tmp = self.faces['F'].stickers[0][ind].color
            self.faces['F'].stickers[0][ind].color = self.faces['R'].stickers[0][ind].color
            self.faces['R'].stickers[0][ind].color = self.faces['B'].stickers[0][ind].color
            self.faces['B'].stickers[0][ind].color = self.faces['L'].stickers[0][ind].color
            self.faces['L'].stickers[0][ind].color = tmp

    def x(self):
        self.swap_faces(['F', 'D', 'B', 'U'])
        self.faces['L'].rotate(1)
        self.faces['R'].rotate(3)

    def y(self):
        self.swap_faces(['F', 'R', 'B', 'L'])
        self.faces['U'].rotate(1)
        self.faces['D'].rotate(3)

    def z(self):
        self.swap_faces(['R', 'U', 'L', 'D'])
        self.faces['F'].rotate(1)
        self.faces['B'].rotate(3)


class CubeEnv(CoreEnv):
    def __init__(self):
        super(CubeEnv, self).__init__()

    def close(self):
        pass

    def render(self, mode: str = 'human'):
        pass

    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    def step(self, action):
        pass


if __name__ == '__main__':
    cube = Cube()
    cube.x()
    cube.draw()

    # points = project_points(
    #     [
    #         (0, 0, -150),
    #         (150, 0, -150),
    #         (150, -150, -150),
    #         (0, -150, -150),
    #         (0, 0, -150)
    #     ],
    #     Quaternion.from_v_theta((1, -1, 0), -np.pi / 6),
    #     (0, 0, 200),
    #     [0, 1, 0]
    # )[:, :2]
    # print(points)
