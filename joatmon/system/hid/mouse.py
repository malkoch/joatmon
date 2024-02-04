import ctypes

from joatmon.core.decorators import auto_pause
from joatmon.system.hid.screen import (
    cursor,
    resolution
)

__all__ = ['Mouse']


@auto_pause(duration=0.05)
def _send_mouse_event(x, y, event):
    """
    Sends a mouse event to the specified coordinates.

    Args:
        x (int): The x-coordinate for the mouse event.
        y (int): The y-coordinate for the mouse event.
        event (int): The type of mouse event.
    """
    height, width = resolution()
    converted_x = 65536 * x // width + 1
    converted_y = 65536 * y // height + 1
    ctypes.windll.user32.mouse_event(event, ctypes.c_long(converted_x), ctypes.c_long(converted_y), 0, 0)


class Mouse:
    """
    A class used to represent a Mouse.

    ...

    Attributes
    ----------
    MOUSE_DOWN : int
        The code for the mouse down event.
    MOUSE_UP : int
        The code for the mouse up event.
    MOUSE_LEFT : int
        The code for the mouse left event.
    MOUSE_RIGHT : int
        The code for the mouse right event.
    MOUSE_MIDDLE : int
        The code for the mouse middle event.
    MIN_X : int
        The minimum x-coordinate for the mouse.
    MAX_X : int
        The maximum x-coordinate for the mouse.
    MIN_Y : int
        The minimum y-coordinate for the mouse.
    MAX_Y : int
        The maximum y-coordinate for the mouse.

    Methods
    -------
    restrict(min_x, max_x, min_y, max_y)
        Restricts the mouse movement to a specific area.
    move_to(x=None, y=None)
        Moves the mouse to the specified coordinates.
    mouse_down(x=None, y=None, button=None)
        Simulates a mouse down event at the specified coordinates.
    mouse_up(x=None, y=None, button=None)
        Simulates a mouse up event at the specified coordinates.
    click(x=None, y=None, button=None)
        Simulates a mouse click event at the specified coordinates.
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
        Restricts the mouse movement to a specific area.

        Args:
            min_x (int): The minimum x-coordinate for the mouse.
            max_x (int): The maximum x-coordinate for the mouse.
            min_y (int): The minimum y-coordinate for the mouse.
            max_y (int): The maximum y-coordinate for the mouse.
        """
        Mouse.min_x = min_x
        Mouse.max_x = max_x
        Mouse.min_y = min_y
        Mouse.max_y = max_y

    @auto_pause(duration=0.05)
    def move_to(self, x=None, y=None):
        """
        Moves the mouse to the specified coordinates.

        Args:
            x (int, optional): The x-coordinate to move the mouse to. If not specified, the mouse's current x-coordinate is used.
            y (int, optional): The y-coordinate to move the mouse to. If not specified, the mouse's current y-coordinate is used.
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
        Simulates a mouse down event at the specified coordinates.

        Args:
            x (int, optional): The x-coordinate for the mouse down event. If not specified, the mouse's current x-coordinate is used.
            y (int, optional): The y-coordinate for the mouse down event. If not specified, the mouse's current y-coordinate is used.
            button (int, optional): The button for the mouse down event. If not specified, the left mouse button is used.
        """
        if button is None:
            button = Mouse.MOUSE_LEFT

        Mouse.move_to(x, y)

        y, x = cursor()
        _send_mouse_event(x, y, button << Mouse.MOUSE_DOWN)

    @staticmethod
    def mouse_up(x=None, y=None, button=None):
        """
        Simulates a mouse up event at the specified coordinates.

        Args:
            x (int, optional): The x-coordinate for the mouse up event. If not specified, the mouse's current x-coordinate is used.
            y (int, optional): The y-coordinate for the mouse up event. If not specified, the mouse's current y-coordinate is used.
            button (int, optional): The button for the mouse up event. If not specified, the left mouse button is used.
        """
        if button is None:
            button = Mouse.MOUSE_LEFT

        Mouse.move_to(x, y)

        y, x = cursor()
        _send_mouse_event(x, y, button << Mouse.MOUSE_UP)

    @staticmethod
    def click(x=None, y=None, button=None):
        """
        Simulates a mouse click event at the specified coordinates.

        Args:
            x (int, optional): The x-coordinate for the mouse click event. If not specified, the mouse's current x-coordinate is used.
            y (int, optional): The y-coordinate for the mouse click event. If not specified, the mouse's current y-coordinate is used.
            button (int, optional): The button for the mouse click event. If not specified, the left mouse button is used.
        """
        if button is None:
            button = Mouse.MOUSE_LEFT

        Mouse.move_to(x, y)

        Mouse.mouse_down(button=button)
        Mouse.mouse_up(button=button)
