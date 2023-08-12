import ctypes
import sys

from joatmon.system.decorators import auto_pause
from joatmon.system.hid.screen import cursor, resolution

__all__ = ['Mouse']

if sys.platform != 'win32':
    raise Exception('The mouse module should only be used on a Windows system.')


@auto_pause(duration=0.05)
def _send_mouse_event(x, y, event):
    height, width = resolution()
    converted_x = 65536 * x // width + 1
    converted_y = 65536 * y // height + 1
    ctypes.windll.user32.mouse_event(event, ctypes.c_long(converted_x), ctypes.c_long(converted_y), 0, 0)


class Mouse:
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

    MOUSE_DOWN = 0x0000
    MOUSE_UP = 0x0001
    MOUSE_LEFT = 0x0002
    MOUSE_RIGHT = 0x0008
    MOUSE_MIDDLE = 0x0020

    MIN_X = None
    MAX_X = None
    MIN_Y = None
    MAX_Y = None

    @staticmethod
    def restrict(min_x, max_x, min_y, max_y):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        Mouse.min_x = min_x
        Mouse.max_x = max_x
        Mouse.min_y = min_y
        Mouse.max_y = max_y

    @auto_pause(duration=0.05)
    def move_to(self, x=None, y=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        # needs a time to move
        if x is None:
            _, x = cursor()
        if y is None:
            y, _ = cursor()

        height, width = resolution()

        min_x = 0
        if Mouse.MIN_X is not None:
            min_x = Mouse.MIN_X

        min_y = 0
        if Mouse.MIN_Y is not None:
            min_y = Mouse.MIN_Y

        max_x = width
        if Mouse.MAX_X is not None:
            max_x = Mouse.MAX_X

        max_y = height
        if Mouse.MAX_Y is not None:
            max_y = Mouse.MAX_Y

        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))

        x = max(min_x, min(x, max_x))
        y = max(min_y, min(y, max_y))

        x = int(x)
        y = int(y)

        ctypes.windll.user32.SetCursorPos(x, y)

    @staticmethod
    def mouse_down(x=None, y=None, button=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if button is None:
            button = Mouse.MOUSE_LEFT

        Mouse.move_to(x, y)

        y, x = cursor()
        _send_mouse_event(x, y, button << Mouse.MOUSE_DOWN)

    @staticmethod
    def mouse_up(x=None, y=None, button=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if button is None:
            button = Mouse.MOUSE_LEFT

        Mouse.move_to(x, y)

        y, x = cursor()
        _send_mouse_event(x, y, button << Mouse.MOUSE_UP)

    @staticmethod
    def click(x=None, y=None, button=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if button is None:
            button = Mouse.MOUSE_LEFT

        Mouse.move_to(x, y)

        Mouse.mouse_down(button=button)
        Mouse.mouse_up(button=button)
