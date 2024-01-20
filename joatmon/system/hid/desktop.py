import collections
import ctypes
import sys
from ctypes import wintypes

__all__ = [
    'RectangleException',
    'GetWindowException',
    'Window',
    'get_windows_at',
    'get_windows_with_title',
    'get_all_titles',
    'get_all_windows',
    'get_window_rect',
    'get_active_window',
    'get_window_text',
]

TOP = 'top'
BOTTOM = 'bottom'
LEFT = 'left'
RIGHT = 'right'
TOP_LEFT = 'top_left'
TOP_RIGHT = 'top_right'
BOTTOM_LEFT = 'bottom_left'
BOTTOM_RIGHT = 'bottom_right'
MID_TOP = 'mid_top'
MID_RIGHT = 'mid_right'
MID_LEFT = 'mid_left'
MID_BOTTOM = 'mid_bottom'
CENTER = 'center'
CENTER_X = 'center_x'
CENTER_Y = 'center_y'
WIDTH = 'width'
HEIGHT = 'height'
SIZE = 'size'
BOX = 'box'
AREA = 'area'

SW_MINIMIZE = 6
SW_MAXIMIZE = 3
SW_RESTORE = 9

HWND_TOP = 0

WM_CLOSE = 0x0010

NULL = 0

FORMAT_MESSAGE_ALLOCATE_BUFFER = 0x00000100
FORMAT_MESSAGE_FROM_SYSTEM = 0x00001000
FORMAT_MESSAGE_IGNORE_INSERTS = 0x00000200

if sys.platform == 'win32':
    EnumWindows = ctypes.windll.user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
    GetWindowText = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible

Rect = collections.namedtuple('Rect', 'left top right bottom')
Box = collections.namedtuple('Box', 'left top width height')

Size = collections.namedtuple('Size', 'width height')
Point = collections.namedtuple('Point', 'x y')


class POINT(ctypes.Structure):
    """
    POINT structure for representing a point in a 2D space.

    Fields:
        x (long): The x-coordinate of the point.
        y (long): The y-coordinate of the point.
    """

    _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]


class RECT(ctypes.Structure):
    """
    RECT structure for representing a rectangle.

    Fields:
        left (long): The x-coordinate of the upper-left corner of the rectangle.
        top (long): The y-coordinate of the upper-left corner of the rectangle.
        right (long): The x-coordinate of the lower-right corner of the rectangle.
        bottom (long): The y-coordinate of the lower-right corner of the rectangle.
    """

    _fields_ = [('left', ctypes.c_long), ('top', ctypes.c_long), ('right', ctypes.c_long), ('bottom', ctypes.c_long)]


class RectangleException(Exception):
    """
    Exception raised for errors in the Rectangle class.
    """


class GetWindowException(Exception):
    """
    Exception raised for errors in getting window information.
    """


def _check_for_int_or_float(arg):
    """
    Check if the argument is an integer or a float.

    Args:
        arg: The argument to check.
    """
    if not isinstance(arg, (int, float)):
        raise RectangleException(f'argument must be int or float, not {arg.__class__.__name__}')


def _check_for_two_int_or_float_tuple(arg):
    """
    Check if the argument is a tuple of two integers or floats.

    Args:
        arg: The argument to check.
    """
    try:
        if not isinstance(arg[0], (int, float)) or not isinstance(arg[1], (int, float)):
            raise RectangleException('argument must be a two-item tuple containing int or float values')
    except Exception as ex:
        print(str(ex))
        raise RectangleException('argument must be a two-item tuple containing int or float values')


def _check_for_four_int_or_float_tuple(arg):
    """
    Check if the argument is a tuple of four integers or floats.

    Args:
        arg: The argument to check.
    """
    try:
        if (
                not isinstance(arg[0], (int, float))
                or not isinstance(arg[1], (int, float))
                or not isinstance(arg[2], (int, float))
                or not isinstance(arg[3], (int, float))
        ):
            raise RectangleException('argument must be a four-item tuple containing int or float values')
    except Exception as ex:
        print(str(ex))
        raise RectangleException('argument must be a four-item tuple containing int or float values')


class Rectangle(object):
    """
    Rectangle class for representing and manipulating a rectangle.

    Methods:
        __init__: Initialize the Rectangle.
        __repr__: Return a string representation of the Rectangle.
        __str__: Return a string representation of the Rectangle.
        call_on_change: Call a function when the Rectangle changes.
        enable_float: Enable or disable floating point values.
        left: Get or set the left coordinate of the Rectangle.
        top: Get or set the top coordinate of the Rectangle.
        right: Get or set the right coordinate of the Rectangle.
        bottom: Get or set the bottom coordinate of the Rectangle.
        top_left: Get or set the top-left coordinate of the Rectangle.
        bottom_left: Get or set the bottom-left coordinate of the Rectangle.
        top_right: Get or set the top-right coordinate of the Rectangle.
        bottom_right: Get or set the bottom-right coordinate of the Rectangle.
        mid_top: Get or set the mid-top coordinate of the Rectangle.
        mid_bottom: Get or set the mid-bottom coordinate of the Rectangle.
        mid_left: Get or set the mid-left coordinate of the Rectangle.
        mid_right: Get or set the mid-right coordinate of the Rectangle.
        center: Get or set the center coordinate of the Rectangle.
        center_x: Get or set the x-coordinate of the center of the Rectangle.
        center_y: Get or set the y-coordinate of the center of the Rectangle.
        size: Get or set the size of the Rectangle.
        width: Get or set the width of the Rectangle.
        height: Get or set the height of the Rectangle.
        area: Get the area of the Rectangle.
        box: Get or set the box of the Rectangle.
        get: Get a property of the Rectangle.
        set: Set a property of the Rectangle.
        move: Move the Rectangle.
        copy: Return a copy of the Rectangle.
        inflate: Inflate the Rectangle.
        clamp: Clamp the Rectangle to another Rectangle.
        union: Union the Rectangle with another Rectangle.
        union_all: Union the Rectangle with a list of other Rectangles.
        normalize: Normalize the Rectangle.
        __contains__: Check if a point or another Rectangle is contained in the Rectangle.
        collide: Check if a point or another Rectangle collides with the Rectangle.
        __eq__: Check if the Rectangle is equal to another Rectangle.
        __ne__: Check if the Rectangle is not equal to another Rectangle.
    """

    def __init__(
            self, left=0, top=0, width=0, height=0, enable_float=False, read_only=False, on_change=None, on_read=None
    ):
        """
        Initialize the Rectangle instance.

        Args:
            left (int, float): The x-coordinate of the left edge of the rectangle. Defaults to 0.
            top (int, float): The y-coordinate of the top edge of the rectangle. Defaults to 0.
            width (int, float): The width of the rectangle. Defaults to 0.
            height (int, float): The height of the rectangle. Defaults to 0.
            enable_float (bool): A flag indicating whether the rectangle's dimensions should be treated as floats. Defaults to False.
            read_only (bool): A flag indicating whether the rectangle's dimensions can be modified. Defaults to False.
            on_change (callable): A function to be called when the rectangle's dimensions change. Must be None or callable. Defaults to None.
            on_read (callable): A function to be called when the rectangle's dimensions are accessed. Must be None or callable. Defaults to None.

        Raises:
            RectangleException: If on_change or on_read is not None or callable.
        """
        _check_for_int_or_float(width)
        _check_for_int_or_float(height)
        _check_for_int_or_float(left)
        _check_for_int_or_float(top)

        self._enable_float = bool(enable_float)
        self._read_only = bool(read_only)

        if on_change is not None and not callable(on_change):
            raise RectangleException('on_change argument must be None or callable (function, method, etc.)')
        self.onChange = on_change

        if on_read is not None and not callable(on_read):
            raise RectangleException('on_read argument must be None or callable (function, method, etc.)')
        self.on_read = on_read

        if enable_float:
            self._width = float(width)
            self._height = float(height)
            self._left = float(left)
            self._top = float(top)
        else:
            self._width = int(width)
            self._height = int(height)
            self._left = int(left)
            self._top = int(top)

    def __repr__(self):
        """
        Returns a string representation of the Rectangle object.
        The string includes the class name and the values of left, top, width, and height attributes.
        """
        return (
            f'{self.__class__.__name__}'
            f'(left={self._left}, '
            f'top={self._top}, '
            f'width={self._width}, '
            f'height={self._height})'
        )

    def __str__(self):
        """
        Returns a string representation of the Rectangle object.
        The string includes the values of left (x), top (y), width (w), and height (h) attributes.
        """
        return f'(x={self._left}, y={self._top}, w={self._width}, h={self._height})'

    def call_on_change(self, old_left, old_top, old_width, old_height):
        """
        Calls the onChange function if it is not None.
        """
        if self.onChange is not None:
            self.onChange(
                Box(old_left, old_top, old_width, old_height), Box(self._left, self._top, self._width, self._height)
            )

    @property
    def enable_float(self):
        """
        Returns the value of the _enable_float attribute.
        """
        return self._enable_float

    @enable_float.setter
    def enable_float(self, value):
        """
        Sets the value of the _enable_float attribute.
        Raises an error if the value is not a boolean.
        """
        if not isinstance(value, bool):
            raise RectangleException('enable_float must be set to a bool value')
        self._enable_float = value

        if self._enable_float:
            self._left = float(self._left)
            self._top = float(self._top)
            self._width = float(self._width)
            self._height = float(self._height)
        else:
            self._left = int(self._left)
            self._top = int(self._top)
            self._width = int(self._width)
            self._height = int(self._height)

    @property
    def left(self):
        """
        Returns the value of the _left attribute.
        """
        if self.on_read is not None:
            self.on_read(LEFT)
        return self._left

    @left.setter
    def left(self, new_left):
        """
        Sets the value of the _left attribute.
        Raises an error if the new value is not an integer or a float.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_int_or_float(new_left)
        if new_left != self._left:
            original_left = self._left
            if self._enable_float:
                self._left = new_left
            else:
                self._left = int(new_left)
            self.call_on_change(original_left, self._top, self._width, self._height)

    @property
    def top(self):
        """
        Returns the value of the _top attribute.
        """
        if self.on_read is not None:
            self.on_read(TOP)
        return self._top

    @top.setter
    def top(self, new_top):
        """
        Sets the value of the _top attribute.
        Raises an error if the new value is not an integer or a float.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_int_or_float(new_top)
        if new_top != self._top:
            original_top = self._top
            if self._enable_float:
                self._top = new_top
            else:
                self._top = int(new_top)
            self.call_on_change(self._left, original_top, self._width, self._height)

    @property
    def right(self):
        """
        Returns the sum of the _left and _width attributes.
        """
        if self.on_read is not None:
            self.on_read(RIGHT)
        return self._left + self._width

    @right.setter
    def right(self, new_right):
        """
        Sets the value of the _left attribute based on the new_right value and the current _width.
        Raises an error if the new value is not an integer or a float.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_int_or_float(new_right)
        if new_right != self._left + self._width:
            original_left = self._left
            if self._enable_float:
                self._left = new_right - self._width
            else:
                self._left = int(new_right) - self._width
            self.call_on_change(original_left, self._top, self._width, self._height)

    @property
    def bottom(self):
        """
        Returns the sum of the _top and _height attributes.
        """
        if self.on_read is not None:
            self.on_read(BOTTOM)
        return self._top + self._height

    @bottom.setter
    def bottom(self, new_bottom):
        """
        Sets the value of the _top attribute based on the new_bottom value and the current _height.
        Raises an error if the new value is not an integer or a float.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_int_or_float(new_bottom)
        if new_bottom != self._top + self._height:
            original_top = self._top
            if self._enable_float:
                self._top = new_bottom - self._height
            else:
                self._top = int(new_bottom) - self._height
            self.call_on_change(self._left, original_top, self._width, self._height)

    @property
    def top_left(self):
        """
        Returns a Point object with the _left and _top attributes as its coordinates.
        """
        if self.on_read is not None:
            self.on_read(TOP_LEFT)
        return Point(x=self._left, y=self._top)

    @top_left.setter
    def top_left(self, value):
        """
        Sets the values of the _left and _top attributes based on the given value.
        Raises an error if the new value is not a tuple of two integers or floats.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_two_int_or_float_tuple(value)
        new_left, new_top = value
        if (new_left != self._left) or (new_top != self._top):
            original_left = self._left
            original_top = self._top
            if self._enable_float:
                self._left = new_left
                self._top = new_top
            else:
                self._left = int(new_left)
                self._top = int(new_top)
            self.call_on_change(original_left, original_top, self._width, self._height)

    @property
    def bottom_left(self):
        """
        Returns a Point object with the _left and the sum of _top and _height as its coordinates.
        """
        if self.on_read is not None:
            self.on_read(BOTTOM_LEFT)
        return Point(x=self._left, y=self._top + self._height)

    @bottom_left.setter
    def bottom_left(self, value):
        """
        Sets the values of the _left and _top attributes based on the given value.
        Raises an error if the new value is not a tuple of two integers or floats.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_two_int_or_float_tuple(value)
        new_left, new_bottom = value
        if (new_left != self._left) or (new_bottom != self._top + self._height):
            original_left = self._left
            original_top = self._top
            if self._enable_float:
                self._left = new_left
                self._top = new_bottom - self._height
            else:
                self._left = int(new_left)
                self._top = int(new_bottom) - self._height
            self.call_on_change(original_left, original_top, self._width, self._height)

    @property
    def top_right(self):
        """
        Returns a Point object with the sum of _left and _width and _top as its coordinates.
        """
        if self.on_read is not None:
            self.on_read(TOP_RIGHT)
        return Point(x=self._left + self._width, y=self._top)

    @top_right.setter
    def top_right(self, value):
        """
        Sets the values of the _left and _top attributes based on the given value.
        Raises an error if the new value is not a tuple of two integers or floats.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_two_int_or_float_tuple(value)
        new_right, new_top = value
        if (new_right != self._left + self._width) or (new_top != self._top):
            original_left = self._left
            original_top = self._top
            if self._enable_float:
                self._left = new_right - self._width
                self._top = new_top
            else:
                self._left = int(new_right) - self._width
                self._top = int(new_top)
            self.call_on_change(original_left, original_top, self._width, self._height)

    @property
    def bottom_right(self):
        """
        Returns a Point object with the sum of _left and _width and the sum of _top and _height as its coordinates.
        """
        if self.on_read is not None:
            self.on_read(BOTTOM_RIGHT)
        return Point(x=self._left + self._width, y=self._top + self._height)

    @bottom_right.setter
    def bottom_right(self, value):
        """
        Sets the values of the _left, _top, _width, and _height attributes based on the given value.
        Raises an error if the new value is not a tuple of two integers or floats.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_two_int_or_float_tuple(value)
        new_right, new_bottom = value
        if (new_bottom != self._top + self._height) or (new_right != self._left + self._width):
            original_left = self._left
            original_top = self._top
            if self._enable_float:
                self._left = new_right - self._width
                self._top = new_bottom - self._height
            else:
                self._left = int(new_right) - self._width
                self._top = int(new_bottom) - self._height
            self.call_on_change(original_left, original_top, self._width, self._height)

    @property
    def mid_top(self):
        """
        Returns a Point object with the middle point of the top edge as its coordinates.
        """
        if self.on_read is not None:
            self.on_read(MID_TOP)
        if self._enable_float:
            return Point(x=self._left + (self._width / 2.0), y=self._top)
        else:
            return Point(x=self._left + (self._width // 2), y=self._top)

    @mid_top.setter
    def mid_top(self, value):
        """
        Sets the values of the _left and _top attributes based on the given value.
        Raises an error if the new value is not a tuple of two integers or floats.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_two_int_or_float_tuple(value)
        new_mid_top, new_top = value
        original_left = self._left
        original_top = self._top
        if self._enable_float:
            if (new_mid_top != self._left + self._width / 2.0) or (new_top != self._top):
                self._left = new_mid_top - (self._width / 2.0)
                self._top = new_top
                self.call_on_change(original_left, original_top, self._width, self._height)
        else:
            if (new_mid_top != self._left + self._width // 2) or (new_top != self._top):
                self._left = int(new_mid_top) - (self._width // 2)
                self._top = int(new_top)
                self.call_on_change(original_left, original_top, self._width, self._height)

    @property
    def mid_bottom(self):
        """
        Returns a Point object with the middle point of the bottom edge as its coordinates.
        """
        if self.on_read is not None:
            self.on_read(MID_BOTTOM)
        if self._enable_float:
            return Point(x=self._left + (self._width / 2.0), y=self._top + self._height)
        else:
            return Point(x=self._left + (self._width // 2), y=self._top + self._height)

    @mid_bottom.setter
    def mid_bottom(self, value):
        """
        Sets the values of the _left and _top attributes based on the given value.
        Raises an error if the new value is not a tuple of two integers or floats.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_two_int_or_float_tuple(value)
        new_mid_bottom, new_bottom = value
        original_left = self._left
        original_top = self._top
        if self._enable_float:
            if (new_mid_bottom != self._left + self._width / 2.0) or (new_bottom != self._top + self._height):
                self._left = new_mid_bottom - (self._width / 2.0)
                self._top = new_bottom - self._height
                self.call_on_change(original_left, original_top, self._width, self._height)
        else:
            if (new_mid_bottom != self._left + self._width // 2) or (new_bottom != self._top + self._height):
                self._left = int(new_mid_bottom) - (self._width // 2)
                self._top = int(new_bottom) - self._height
                self.call_on_change(original_left, original_top, self._width, self._height)

    @property
    def mid_left(self):
        """
        Returns a Point object with the middle point of the left edge as its coordinates.
        """
        if self.on_read is not None:
            self.on_read(MID_LEFT)
        if self._enable_float:
            return Point(x=self._left, y=self._top + (self._height / 2.0))
        else:
            return Point(x=self._left, y=self._top + (self._height // 2))

    @mid_left.setter
    def mid_left(self, value):
        """
        Sets the values of the _left and _top attributes based on the given value.
        Raises an error if the new value is not a tuple of two integers or floats.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_two_int_or_float_tuple(value)
        new_left, new_mid_left = value
        original_left = self._left
        original_top = self._top
        if self._enable_float:
            if (new_left != self._left) or (new_mid_left != self._top + (self._height / 2.0)):
                self._left = new_left
                self._top = new_mid_left - (self._height / 2.0)
                self.call_on_change(original_left, original_top, self._width, self._height)
        else:
            if (new_left != self._left) or (new_mid_left != self._top + (self._height // 2)):
                self._left = int(new_left)
                self._top = int(new_mid_left) - (self._height // 2)
                self.call_on_change(original_left, original_top, self._width, self._height)

    @property
    def mid_right(self):
        """
        Returns a Point object with the middle point of the right edge as its coordinates.
        """
        if self.on_read is not None:
            self.on_read(MID_RIGHT)
        if self._enable_float:
            return Point(x=self._left + self._width, y=self._top + (self._height / 2.0))
        else:
            return Point(x=self._left + self._width, y=self._top + (self._height // 2))

    @mid_right.setter
    def mid_right(self, value):
        """
        Sets the values of the _left and _top attributes based on the given value.
        Raises an error if the new value is not a tuple of two integers or floats.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_two_int_or_float_tuple(value)
        new_right, new_mid_right = value
        original_left = self._left
        original_top = self._top
        if self._enable_float:
            if (new_right != self._left + self._width) or (new_mid_right != self._top + self._height / 2.0):
                self._left = new_right - self._width
                self._top = new_mid_right - (self._height / 2.0)
                self.call_on_change(original_left, original_top, self._width, self._height)
        else:
            if (new_right != self._left + self._width) or (new_mid_right != self._top + self._height // 2):
                self._left = int(new_right) - self._width
                self._top = int(new_mid_right) - (self._height // 2)
                self.call_on_change(original_left, original_top, self._width, self._height)

    @property
    def center(self):
        """
        Returns a Point object with the center point of the rectangle as its coordinates.
        """
        if self.on_read is not None:
            self.on_read(CENTER)
        if self._enable_float:
            return Point(x=self._left + (self._width / 2.0), y=self._top + (self._height / 2.0))
        else:
            return Point(x=self._left + (self._width // 2), y=self._top + (self._height // 2))

    @center.setter
    def center(self, value):
        """
        Sets the values of the _left and _top attributes based on the given value.
        Raises an error if the new value is not a tuple of two integers or floats.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_two_int_or_float_tuple(value)
        new_center_x, new_center_y = value
        original_left = self._left
        original_top = self._top
        if self._enable_float:
            if (new_center_x != self._left + self._width / 2.0) or (new_center_y != self._top + self._height / 2.0):
                self._left = new_center_x - (self._width / 2.0)
                self._top = new_center_y - (self._height / 2.0)
                self.call_on_change(original_left, original_top, self._width, self._height)
        else:
            if (new_center_x != self._left + self._width // 2) or (new_center_y != self._top + self._height // 2):
                self._left = int(new_center_x) - (self._width // 2)
                self._top = int(new_center_y) - (self._height // 2)
                self.call_on_change(original_left, original_top, self._width, self._height)

    @property
    def center_x(self):
        """
        Returns the x-coordinate of the center point of the rectangle.
        """
        if self.on_read is not None:
            self.on_read(CENTER_X)
        if self._enable_float:
            return self._left + (self._width / 2.0)
        else:
            return self._left + (self._width // 2)

    @center_x.setter
    def center_x(self, new_center_x):
        """
        Sets the value of the _left attribute based on the given value.
        Raises an error if the new value is not an integer or a float.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_int_or_float(new_center_x)
        original_left = self._left
        if self._enable_float:
            if new_center_x != self._left + self._width / 2.0:
                self._left = new_center_x - (self._width / 2.0)
                self.call_on_change(original_left, self._top, self._width, self._height)
        else:
            if new_center_x != self._left + self._width // 2:
                self._left = int(new_center_x) - (self._width // 2)
                self.call_on_change(original_left, self._top, self._width, self._height)

    @property
    def center_y(self):
        """
        Returns the y-coordinate of the center point of the rectangle.
        """
        if self.on_read is not None:
            self.on_read(CENTER_Y)
        if self._enable_float:
            return self._top + (self._height / 2.0)
        else:
            return self._top + (self._height // 2)

    @center_y.setter
    def center_y(self, new_center_y):
        """
        Sets the value of the _top attribute based on the given value.
        Raises an error if the new value is not an integer or a float.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_int_or_float(new_center_y)
        original_top = self._top
        if self._enable_float:
            if new_center_y != self._top + self._height / 2.0:
                self._top = new_center_y - (self._height / 2.0)
                self.call_on_change(self._left, original_top, self._width, self._height)
        else:
            if new_center_y != self._top + self._height // 2:
                self._top = int(new_center_y) - (self._height // 2)
                self.call_on_change(self._left, original_top, self._width, self._height)

    @property
    def size(self):
        """
        Returns a Size object with the _width and _height attributes as its dimensions.
        """
        if self.on_read is not None:
            self.on_read(SIZE)
        return Size(width=self._width, height=self._height)

    @size.setter
    def size(self, value):
        """
        Sets the values of the _width and _height attributes based on the given value.
        Raises an error if the new value is not a tuple of two integers or floats.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_two_int_or_float_tuple(value)
        new_width, new_height = value
        if new_width != self._width or new_height != self._height:
            original_width = self._width
            original_height = self._height
            if self._enable_float:
                self._width = new_width
                self._height = new_height
            else:
                self._width = int(new_width)
                self._height = int(new_height)
            self.call_on_change(self._left, self._top, original_width, original_height)

    @property
    def width(self):
        """
        Returns the value of the _width attribute.
        """
        if self.on_read is not None:
            self.on_read(WIDTH)
        return self._width

    @width.setter
    def width(self, new_width):
        """
        Sets the value of the _width attribute.
        Raises an error if the new value is not an integer or a float.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_int_or_float(new_width)
        if new_width != self._width:
            original_width = self._width
            if self._enable_float:
                self._width = new_width
            else:
                self._width = int(new_width)
            self.call_on_change(self._left, self._top, original_width, self._height)

    @property
    def height(self):
        """
        Property getter for _height attribute of the Rectangle instance.

        Returns:
            int or float: The height of the rectangle.
        """
        if self.on_read is not None:
            self.on_read(HEIGHT)
        return self._height

    @height.setter
    def height(self, new_height):
        """
        Property setter for _height attribute of the Rectangle instance.

        Args:
            new_height (int or float): The new height of the rectangle.

        Raises:
            RectangleException: If the Rectangle instance is read-only.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_int_or_float(new_height)
        if new_height != self._height:
            original_height = self._height
            if self._enable_float:
                self._height = new_height
            else:
                self._height = int(new_height)
            self.call_on_change(self._left, self._top, self._width, original_height)

    x = left
    y = top
    w = width
    h = height

    @property
    def area(self):
        """
        Property getter for the area of the Rectangle instance.

        Returns:
            int or float: The area of the rectangle.
        """
        if self.on_read is not None:
            self.on_read(AREA)
        return self._width * self._height

    @property
    def box(self):
        """
        Property getter for the box representation of the Rectangle instance.

        Returns:
            Box: The box representation of the rectangle.
        """
        if self.on_read is not None:
            self.on_read(BOX)
        return Box(left=self._left, top=self._top, width=self._width, height=self._height)

    @box.setter
    def box(self, value):
        """
        Property setter for the box representation of the Rectangle instance.

        Args:
            value (tuple): The new box representation of the rectangle.

        Raises:
            RectangleException: If the Rectangle instance is read-only.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_four_int_or_float_tuple(value)
        new_left, new_top, new_width, new_height = value
        if (
                (new_left != self._left)
                or (new_top != self._top)
                or (new_width != self._width)
                or (new_height != self._height)
        ):
            original_left = self._left
            original_top = self._top
            original_width = self._width
            original_height = self._height
            if self._enable_float:
                self._left = float(new_left)
                self._top = float(new_top)
                self._width = float(new_width)
                self._height = float(new_height)
            else:
                self._left = int(new_left)
                self._top = int(new_top)
                self._width = int(new_width)
                self._height = int(new_height)
            self.call_on_change(original_left, original_top, original_width, original_height)

    def get(self, rect_attr_name):
        """
        Returns the value of the specified rectangle attribute.

        Args:
            rect_attr_name (str): The name of the rectangle attribute.

        Returns:
            int or float: The value of the specified rectangle attribute.

        Raises:
            RectangleException: If the attribute name is not valid.
        """
        if rect_attr_name == TOP:
            return self.top
        elif rect_attr_name == BOTTOM:
            return self.bottom
        elif rect_attr_name == LEFT:
            return self.left
        elif rect_attr_name == RIGHT:
            return self.right
        elif rect_attr_name == TOP_LEFT:
            return self.top_left
        elif rect_attr_name == TOP_RIGHT:
            return self.top_right
        elif rect_attr_name == BOTTOM_LEFT:
            return self.bottom_left
        elif rect_attr_name == BOTTOM_RIGHT:
            return self.bottom_right
        elif rect_attr_name == MID_TOP:
            return self.mid_top
        elif rect_attr_name == MID_BOTTOM:
            return self.mid_bottom
        elif rect_attr_name == MID_LEFT:
            return self.mid_left
        elif rect_attr_name == MID_RIGHT:
            return self.mid_right
        elif rect_attr_name == CENTER:
            return self.center
        elif rect_attr_name == CENTER_X:
            return self.center_x
        elif rect_attr_name == CENTER_Y:
            return self.center_y
        elif rect_attr_name == WIDTH:
            return self.width
        elif rect_attr_name == HEIGHT:
            return self.height
        elif rect_attr_name == SIZE:
            return self.size
        elif rect_attr_name == AREA:
            return self.area
        elif rect_attr_name == BOX:
            return self.box
        else:
            raise RectangleException(f"'{rect_attr_name}' is not a valid attribute name")

    def set(self, rect_attr_name, value):
        """
        Sets the value of the specified rectangle attribute.

        Args:
            rect_attr_name (str): The name of the rectangle attribute.
            value (int or float): The new value of the rectangle attribute.

        Raises:
            RectangleException: If the attribute name is not valid or if the attribute is read-only.
        """
        if rect_attr_name == TOP:
            self.top = value
        elif rect_attr_name == BOTTOM:
            self.bottom = value
        elif rect_attr_name == LEFT:
            self.left = value
        elif rect_attr_name == RIGHT:
            self.right = value
        elif rect_attr_name == TOP_LEFT:
            self.top_left = value
        elif rect_attr_name == TOP_RIGHT:
            self.top_right = value
        elif rect_attr_name == BOTTOM_LEFT:
            self.bottom_left = value
        elif rect_attr_name == BOTTOM_RIGHT:
            self.bottom_right = value
        elif rect_attr_name == MID_TOP:
            self.mid_top = value
        elif rect_attr_name == MID_BOTTOM:
            self.mid_bottom = value
        elif rect_attr_name == MID_LEFT:
            self.mid_left = value
        elif rect_attr_name == MID_RIGHT:
            self.mid_right = value
        elif rect_attr_name == CENTER:
            self.center = value
        elif rect_attr_name == CENTER_X:
            self.center_x = value
        elif rect_attr_name == CENTER_Y:
            self.center_y = value
        elif rect_attr_name == WIDTH:
            self.width = value
        elif rect_attr_name == HEIGHT:
            self.height = value
        elif rect_attr_name == SIZE:
            self.size = value
        elif rect_attr_name == AREA:
            raise RectangleException('area is a read-only attribute')
        elif rect_attr_name == BOX:
            self.box = value
        else:
            raise RectangleException(f"'{rect_attr_name}' is not a valid attribute name")

    def move(self, x_offset, y_offset):
        """
        Moves the rectangle by the specified offsets.

        Args:
            x_offset (int or float): The offset along the x-axis.
            y_offset (int or float): The offset along the y-axis.

        Raises:
            RectangleException: If the Rectangle instance is read-only.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        _check_for_int_or_float(x_offset)
        _check_for_int_or_float(y_offset)
        if self._enable_float:
            self._left += x_offset
            self._top += y_offset
        else:
            self._left += int(x_offset)
            self._top += int(y_offset)

    def copy(self):
        """
        Returns a copy of the Rectangle instance.

        Returns:
            Rectangle: A copy of the Rectangle instance.
        """
        return Rectangle(self._left, self._top, self._width, self._height, self._enable_float, self._read_only)

    def inflate(self, width_change=0, height_change=0):
        """
        Increases the size of the rectangle by the specified amounts.

        Args:
            width_change (int or float): The change in width. Defaults to 0.
            height_change (int or float): The change in height. Defaults to 0.

        Raises:
            RectangleException: If the Rectangle instance is read-only.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        original_center = self.center
        self.width += width_change
        self.height += height_change
        self.center = original_center

    def clamp(self, other_rect):
        """
        Moves the rectangle to be within the bounds of another rectangle.

        Args:
            other_rect (Rectangle): The other rectangle.

        Raises:
            RectangleException: If the Rectangle instance is read-only.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        self.center = other_rect.center

    def union(self, other_rect):
        """
        Expands the rectangle to include another rectangle.

        Args:
            other_rect (Rectangle): The other rectangle.
        """
        union_left = min(self._left, other_rect._left)
        union_top = min(self._top, other_rect._top)
        union_right = max(self.right, other_rect.right)
        union_bottom = max(self.bottom, other_rect.bottom)

        self._left = union_left
        self._top = union_top
        self._width = union_right - union_left
        self._height = union_bottom - union_top

    def union_all(self, other_rectangles):
        """
        Expands the rectangle to include a list of other rectangles.

        Args:
            other_rectangles (list): The list of other rectangles.
        """
        other_rectangles = list(other_rectangles)
        other_rectangles.append(self)

        union_left = min([r._left for r in other_rectangles])
        union_top = min([r._top for r in other_rectangles])
        union_right = max([r.right for r in other_rectangles])
        union_bottom = max([r.bottom for r in other_rectangles])

        self._left = union_left
        self._top = union_top
        self._width = union_right - union_left
        self._height = union_bottom - union_top

    def normalize(self):
        """
        Adjusts the rectangle to have non-negative width and height.

        Raises:
            RectangleException: If the Rectangle instance is read-only.
        """
        if self._read_only:
            raise RectangleException('Rect object is read-only')

        if self._width < 0:
            self._width = -self._width
            self._left -= self._width
        if self._height < 0:
            self._height = -self._height
            self._top -= self._height

    def __contains__(self, value):
        """
        Checks whether a point or rectangle is contained within the rectangle.

        Args:
            value (tuple or Rectangle): The point or rectangle to check.

        Returns:
            bool: True if the point or rectangle is contained within the rectangle, False otherwise.

        Raises:
            RectangleException: If the value is not a valid point or rectangle.
        """
        if isinstance(value, Rectangle):
            return (
                    value.top_left in self
                    and value.top_right in self
                    and value.bottom_left in self
                    and value.bottom_right in self
            )

        try:
            len(value)
        except Exception as ex:
            print(str(ex))
            raise RectangleException(
                f'in <Rect> requires an (x, y) tuple, '
                f'a (left, top, width, height) tuple, or a Rect object as '
                f'left operand, not {value.__class__.__name__}'
            )

        if len(value) == 2:
            _check_for_two_int_or_float_tuple(value)
            x, y = value
            return self._left < x < self._left + self._width and self._top < y < self._top + self._height
        elif len(value) == 4:
            _check_for_four_int_or_float_tuple(value)
            left, top, width, height = value
            return (
                    (left, top) in self
                    and (left + width, top) in self
                    and (left, top + height) in self
                    and (left + width, top + height) in self
            )
        else:
            raise RectangleException(
                f'in <Rect> requires an (x, y) tuple, a'
                f' (left, top, width, height) tuple, or a Rect object as '
                f'left operand, not {value.__class__.__name__}'
            )

    def collide(self, value):
        """
        Checks whether a point or rectangle collides with the rectangle.

        Args:
            value (tuple or Rectangle): The point or rectangle to check.

        Returns:
            bool: True if the point or rectangle collides with the rectangle, False otherwise.

        Raises:
            RectangleException: If the value is not a valid point or rectangle.
        """
        if isinstance(value, Rectangle):
            return (
                    value.top_left in self
                    or value.top_right in self
                    or value.bottom_left in self
                    or value.bottom_right in self
            )

        try:
            len(value)
        except Exception as ex:
            print(str(ex))
            raise RectangleException(
                f'in <Rect> requires an (x, y) tuple, a '
                f'(left, top, width, height) tuple, or a Rect object as '
                f'left operand, not {value.__class__.__name__}'
            )

        if len(value) == 2:
            _check_for_two_int_or_float_tuple(value)
            x, y = value
            return self._left < x < self._left + self._width and self._top < y < self._top + self._height
        elif len(value) == 4:
            left, top, width, height = value
            return (
                    (left, top) in self
                    or (left + width, top) in self
                    or (left, top + height) in self
                    or (left + width, top + height) in self
            )
        else:
            raise RectangleException(
                f'in <Rect> requires an (x, y) tuple, '
                f'a (left, top, width, height) tuple, or a Rect object as '
                f'left operand, not {value.__class__.__name__}'
            )

    def __eq__(self, other):
        """
        Checks whether the rectangle is equal to another rectangle.

        Args:
            other (Rectangle): The rectangle to compare with.

        Returns:
            bool: True if the rectangles are equal, False otherwise.

        Raises:
            RectangleException: If the other object is not a Rectangle.
        """
        if isinstance(other, Rectangle):
            return other.box == self.box
        else:
            raise RectangleException('Rect objects can only be compared with other Rect objects')

    def __ne__(self, other):
        """
        Checks whether the rectangle is not equal to another rectangle.

        Args:
            other (Rectangle): The rectangle to compare with.

        Returns:
            bool: True if the rectangles are not equal, False otherwise.

        Raises:
            RectangleException: If the other object is not a Rectangle.
        """
        if isinstance(other, Rectangle):
            return other.box != self.box
        else:
            raise RectangleException('Rect objects can only be compared with other Rect objects')


def _format_message(error_code):
    """
    Format a Windows error message.

    Args:
        error_code (int): The error code to format.

    Returns:
        str: The formatted error message.
    """
    lp_buffer = wintypes.LPWSTR()

    ctypes.windll.kernel32.FormatMessageW(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        error_code,
        0,
        ctypes.cast(ctypes.byref(lp_buffer), wintypes.LPWSTR),
        0,
        NULL,
    )
    msg = lp_buffer.value.rstrip()
    ctypes.windll.kernel32.LocalFree(lp_buffer)
    return msg


def _raise_with_last_error():
    """
    Raise an exception with the last Windows error message.
    """
    error_code = ctypes.windll.kernel32.GetLastError()
    raise GetWindowException(f'Error code from Windows: {error_code} - {_format_message(error_code)}')


def point_in_rect(x, y, left, top, width, height):
    """
    Check if a point is in a rectangle.

    Args:
        x (int): The x-coordinate of the point.
        y (int): The y-coordinate of the point.
        left (int): The left coordinate of the rectangle.
        top (int): The top coordinate of the rectangle.
        width (int): The width of the rectangle.
        height (int): The height of the rectangle.

    Returns:
        bool: True if the point is in the rectangle, False otherwise.
    """
    return left < x < left + width and top < y < top + height


def _get_all_titles():
    """
    Get the titles of all visible windows.

    Returns:
        list: A list of window titles.
    """
    titles = []

    def foreach_window(h_wnd, _):
        if IsWindowVisible(h_wnd):
            length = GetWindowTextLength(h_wnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            GetWindowText(h_wnd, buff, length + 1)
            titles.append((h_wnd, buff.value))
        return True

    EnumWindows(EnumWindowsProc(foreach_window), 0)

    return titles


def get_windows_at(x, y):
    """
    Get all windows at a specific point.

    Args:
        x (int): The x-coordinate of the point.
        y (int): The y-coordinate of the point.

    Returns:
        list: A list of windows at the point.
    """
    windows_at_xy = []
    for _window in get_all_windows():
        if point_in_rect(x, y, _window.left, _window.top, _window.width, _window.height):
            windows_at_xy.append(_window)
    return windows_at_xy


def get_windows_with_title(title):
    """
    Get all windows with a specific title.

    Args:
        title (str): The title to search for.

    Returns:
        list: A list of windows with the title.
    """
    h_windows_and_titles = _get_all_titles()
    window_objects = []
    for hWnd, winTitle in h_windows_and_titles:
        if title.upper() in winTitle.upper():
            window_objects.append(Window(hWnd))
    return window_objects


def get_all_titles():
    """
    Get the titles of all windows.

    Returns:
        list: A list of window titles.
    """
    return [_window.title for _window in get_all_windows()]


def get_all_windows():
    """
    Get all windows.

    Returns:
        list: A list of all windows.
    """
    window_objects = []

    def foreach_window(h_wnd, _):
        if ctypes.windll.user32.IsWindowVisible(h_wnd) != 0:
            window_objects.append(Window(h_wnd))
        return True

    EnumWindows(EnumWindowsProc(foreach_window), 0)

    return window_objects


def get_window_rect(h_wnd):
    """
    Get the rectangle of a window.

    Args:
        h_wnd (int): The handle of the window.

    Returns:
        Rect: The rectangle of the window.
    """
    _rect = RECT()
    result = ctypes.windll.user32.GetWindowRect(h_wnd, ctypes.byref(_rect))
    if result != 0:
        return Rect(_rect.left, _rect.top, _rect.right, _rect.bottom)
    else:
        _raise_with_last_error()


def get_active_window():
    """
    Get the active window.

    Returns:
        Window: The active window.
    """
    h_wnd = ctypes.windll.user32.GetForegroundWindow()
    if h_wnd == 0:
        return None
    else:
        return Window(h_wnd)


def get_window_text(h_wnd):
    """
    Get the text of a window.

    Args:
        h_wnd (int): The handle of the window.

    Returns:
        str: The text of the window.
    """
    text_len_in_characters = ctypes.windll.user32.GetWindowTextLengthW(h_wnd)
    string_buffer = ctypes.create_unicode_buffer(text_len_in_characters + 1)
    ctypes.windll.user32.GetWindowTextW(h_wnd, string_buffer, text_len_in_characters + 1)
    return string_buffer.value


class Window:
    """
    Window class for representing and manipulating a window.

    Methods:
        __init__: Initialize the Window.
        __repr__: Return a string representation of the Window.
        __str__: Return a string representation of the Window.
        __eq__: Check if the Window is equal to another Window.
        close: Close the Window.
        minimize: Minimize the Window.
        maximize: Maximize the Window.
        restore: Restore the Window.
        activate: Activate the Window.
        resize_rel: Resize the Window relative to its current size.
        resize_to: Resize the Window to a specific size.
        move_rel: Move the Window relative to its current position.
        move_to: Move the Window to a specific position.
        h_wnd: Get the handle of the Window.
        is_minimized: Check if the Window is minimized.
        is_maximized: Check if the Window is maximized.
        is_active: Check if the Window is active.
        title: Get the title of the Window.
        visible: Check if the Window is visible.
        left: Get or set the left coordinate of the Window.
        right: Get or set the right coordinate of the Window.
        top: Get or set the top coordinate of the Window.
        bottom: Get or set the bottom coordinate of the Window.
        top_left: Get or set the top-left coordinate of the Window.
        bottom_left: Get or set the bottom-left coordinate of the Window.
        top_right: Get or set the top-right coordinate of the Window.
        bottom_right: Get or set the bottom-right coordinate of the Window.
        mid_left: Get or set the mid-left coordinate of the Window.
        mid_right: Get or set the mid-right coordinate of the Window.
        mid_top: Get or set the mid-top coordinate of the Window.
        mid_bottom: Get or set the mid-bottom coordinate of the Window.
        center: Get or set the center coordinate of the Window.
        center_x: Get or set the x-coordinate of the center of the Window.
        center_y: Get or set the y-coordinate of the center of the Window.
        width: Get or set the width of the Window.
        height: Get or set the height of the Window.
        size: Get or set the size of the Window.
        area: Get the area of the Window.
        box: Get or set the box of the Window.
    """

    def __init__(self, h_wnd):
        """
        Initializes a new instance of the Window class.

        Args:
            h_wnd: The handle to the window.
        """
        self._h_wnd = h_wnd

        def _on_read(_):
            """
            Updates the rectangle's position and size when the window's position or size is read.
            """
            _rect = get_window_rect(self._h_wnd)
            self._rect._left = _rect.left
            self._rect._top = _rect.top
            self._rect._width = _rect.right - _rect.left
            self._rect._height = _rect.bottom - _rect.top

        def _on_change(_, new_box):
            """
            Updates the window's position and size when the rectangle's position or size is changed.

            Args:
                new_box: The new position and size of the rectangle.
            """
            self.move_to(new_box.left, new_box.top)
            self.resize_to(new_box.width, new_box.height)

        r = get_window_rect(self._h_wnd)
        self._rect = Rectangle(
            r.left, r.top, r.right - r.left, r.bottom - r.top, on_change=_on_change, on_read=_on_read
        )

    def __str__(self):
        """
        Returns a string representation of the Window instance.

        Returns:
            A string representation of the Window instance.
        """
        r = get_window_rect(self._h_wnd)
        width = r.right - r.left
        height = r.bottom - r.top
        return (
            f'<{self.__class__.__name__} '
            f'left="{r.left}", '
            f'top="{r.top}", '
            f'width="{width}", '
            f'height="{height}", '
            f'title="{self.title}">'
        )

    def __repr__(self):
        """
        Returns a string that can be used to recreate the Window instance.

        Returns:
            A string that can be used to recreate the Window instance.
        """
        return '%s(hWnd=%s)' % (self.__class__.__name__, self._h_wnd)

    def __eq__(self, other):
        """
        Checks whether the Window instance is equal to another Window instance.

        Args:
            other: The other Window instance to compare with.

        Returns:
            True if the Window instance is equal to the other Window instance, False otherwise.
        """
        return isinstance(other, Window) and self._h_wnd == other._h_wnd

    def close(self):
        """
        Closes the window.
        """
        result = ctypes.windll.user32.PostMessageA(self._h_wnd, WM_CLOSE, 0, 0)
        if result == 0:
            _raise_with_last_error()

    def minimize(self):
        """
        Minimizes the window.
        """
        ctypes.windll.user32.ShowWindow(self._h_wnd, SW_MINIMIZE)

    def maximize(self):
        """
        Maximizes the window.
        """
        ctypes.windll.user32.ShowWindow(self._h_wnd, SW_MAXIMIZE)

    def restore(self):
        """
        Restores the window to its original size and position.
        """
        ctypes.windll.user32.ShowWindow(self._h_wnd, SW_RESTORE)

    def activate(self):
        """
        Brings the window to the foreground.
        """
        result = ctypes.windll.user32.SetForegroundWindow(self._h_wnd)
        if result == 0:
            _raise_with_last_error()

    def resize_rel(self, width_offset, height_offset):
        """
        Resizes the window relative to its current size.

        Args:
            width_offset: The offset to add to the window's current width.
            height_offset: The offset to add to the window's current height.
        """
        result = ctypes.windll.user32.SetWindowPos(
            self._h_wnd, HWND_TOP, self.left, self.top, self.width + width_offset, self.height + height_offset, 0
        )
        if result == 0:
            _raise_with_last_error()

    def resize_to(self, new_width, new_height):
        """
        Resizes the window to a specific size.

        Args:
            new_width: The new width of the window.
            new_height: The new height of the window.
        """
        result = ctypes.windll.user32.SetWindowPos(self._h_wnd, HWND_TOP, self.left, self.top, new_width, new_height, 0)
        if result == 0:
            _raise_with_last_error()

    def move_rel(self, x_offset, y_offset):
        """
        Moves the window relative to its current position.

        Args:
            x_offset: The offset to add to the window's current x-coordinate.
            y_offset: The offset to add to the window's current y-coordinate.
        """
        result = ctypes.windll.user32.SetWindowPos(
            self._h_wnd, HWND_TOP, self.left + x_offset, self.top + y_offset, self.width, self.height, 0
        )
        if result == 0:
            _raise_with_last_error()

    def move_to(self, new_left, new_top):
        """
        Moves the window to a specific position.

        Args:
            new_left: The new left position of the window.
            new_top: The new top position of the window.
        """
        result = ctypes.windll.user32.SetWindowPos(self._h_wnd, HWND_TOP, new_left, new_top, self.width, self.height, 0)
        if result == 0:
            _raise_with_last_error()

    @property
    def h_wnd(self):
        """
        Gets the handle to the window.

        Returns:
            The handle to the window.
        """
        return self._h_wnd

    @property
    def is_minimized(self):
        """
        Checks whether the window is minimized.

        Returns:
            True if the window is minimized, False otherwise.
        """
        return ctypes.windll.user32.IsIconic(self._h_wnd) != 0

    @property
    def is_maximized(self):
        """
        Checks whether the window is maximized.

        Returns:
            True if the window is maximized, False otherwise.
        """
        return ctypes.windll.user32.IsZoomed(self._h_wnd) != 0

    @property
    def is_active(self):
        """
        Checks whether the window is the active window.

        Returns:
            True if the window is the active window, False otherwise.
        """
        return get_active_window() == self

    @property
    def title(self):
        """
        Gets the title of the window.

        Returns:
            The title of the window.
        """
        return get_window_text(self._h_wnd)

    @property
    def visible(self):
        """
        Checks whether the window is visible.

        Returns:
            True if the window is visible, False otherwise.
        """
        return IsWindowVisible(self._h_wnd)

    @property
    def left(self):
        """
        Gets the left position of the window.

        Returns:
            The left position of the window.
        """
        return self._rect.left

    @left.setter
    def left(self, value):
        """
        Sets the left position of the window.

        Args:
            value: The new left position of the window.
        """
        _ = self._rect.left
        self._rect.left = value

    @property
    def right(self):
        """
        Gets the right position of the window.

        Returns:
            The right position of the window.
        """
        return self._rect.right

    @right.setter
    def right(self, value):
        """
        Sets the right position of the window.

        Args:
            value: The new right position of the window.
        """
        _ = self._rect.right
        self._rect.right = value

    @property
    def top(self):
        """
        Gets the top position of the window.

        Returns:
            The top position of the window.
        """
        return self._rect.top

    @top.setter
    def top(self, value):
        """
        Sets the top position of the window.

        Args:
            value: The new top position of the window.
        """
        _ = self._rect.top
        self._rect.top = value

    @property
    def bottom(self):
        """
        Gets the bottom position of the window.

        Returns:
            The bottom position of the window.
        """
        return self._rect.bottom

    @bottom.setter
    def bottom(self, value):
        """
        Sets the bottom position of the window.

        Args:
            value: The new bottom position of the window.
        """
        _ = self._rect.bottom
        self._rect.bottom = value

    @property
    def top_left(self):
        """
        Gets the top-left position of the window.

        Returns:
            The top-left position of the window.
        """
        return self._rect.top_left

    @top_left.setter
    def top_left(self, value):
        """
        Sets the top-left position of the window.

        Args:
            value: The new top-left position of the window.
        """
        _ = self._rect.top_left
        self._rect.top_left = value

    @property
    def top_right(self):
        """
        Gets the top-right position of the window.

        Returns:
            The top-right position of the window.
        """
        return self._rect.top_right

    @top_right.setter
    def top_right(self, value):
        """
        Sets the top-right position of the window.

        Args:
            value: The new top-right position of the window.
        """
        _ = self._rect.top_right
        self._rect.top_right = value

    @property
    def bottom_left(self):
        """
        Gets the bottom-left position of the window.

        Returns:
            The bottom-left position of the window.
        """
        return self._rect.bottom_left

    @bottom_left.setter
    def bottom_left(self, value):
        """
        Sets the bottom-left position of the window.

        Args:
            value: The new bottom-left position of the window.
        """
        _ = self._rect.bottom_left
        self._rect.bottom_left = value

    @property
    def bottom_right(self):
        """
        Gets the bottom-right position of the window.

        Returns:
            The bottom-right position of the window.
        """
        return self._rect.bottom_right

    @bottom_right.setter
    def bottom_right(self, value):
        """
        Sets the bottom-right position of the window.

        Args:
            value: The new bottom-right position of the window.
        """
        _ = self._rect.bottom_right
        self._rect.bottom_right = value

    @property
    def mid_left(self):
        """
        Gets the middle-left position of the window.

        Returns:
            The middle-left position of the window.
        """
        return self._rect.mid_left

    @mid_left.setter
    def mid_left(self, value):
        """
        Sets the middle-left position of the window.

        Args:
            value: The new middle-left position of the window.
        """
        _ = self._rect.mid_left
        self._rect.mid_left = value

    @property
    def mid_right(self):
        """
        Gets the middle-right position of the window.

        Returns:
            The middle-right position of the window.
        """
        return self._rect.mid_right

    @mid_right.setter
    def mid_right(self, value):
        """
        Setter for the mid_right property of the Window instance.

        Args:
            value (tuple): A tuple representing the new mid-right coordinates of the window.
        """
        _ = self._rect.mid_right
        self._rect.mid_right = value

    @property
    def mid_top(self):
        """
        Getter for the mid_top property of the Window instance.

        Returns:
            tuple: The mid-top coordinates of the window.
        """
        return self._rect.mid_top

    @mid_top.setter
    def mid_top(self, value):
        """
        Setter for the mid_top property of the Window instance.

        Args:
            value (tuple): A tuple representing the new mid-top coordinates of the window.
        """
        _ = self._rect.mid_top
        self._rect.mid_top = value

    @property
    def mid_bottom(self):
        """
        Getter for the mid_bottom property of the Window instance.

        Returns:
            tuple: The mid-bottom coordinates of the window.
        """
        return self._rect.mid_bottom

    @mid_bottom.setter
    def mid_bottom(self, value):
        """
        Setter for the mid_bottom property of the Window instance.

        Args:
            value (tuple): A tuple representing the new mid-bottom coordinates of the window.
        """
        _ = self._rect.mid_bottom
        self._rect.mid_bottom = value

    @property
    def center(self):
        """
        Getter for the center property of the Window instance.

        Returns:
            tuple: The center coordinates of the window.
        """
        return self._rect.center

    @center.setter
    def center(self, value):
        """
        Setter for the center property of the Window instance.

        Args:
            value (tuple): A tuple representing the new center coordinates of the window.
        """
        _ = self._rect.center
        self._rect.center = value

    @property
    def center_x(self):
        """
        Getter for the center_x property of the Window instance.

        Returns:
            int: The x-coordinate of the center of the window.
        """
        return self._rect.center_x

    @center_x.setter
    def center_x(self, value):
        """
        Setter for the center_x property of the Window instance.

        Args:
            value (int): The new x-coordinate of the center of the window.
        """
        _ = self._rect.center_x
        self._rect.center_x = value

    @property
    def center_y(self):
        """
        Getter for the center_y property of the Window instance.

        Returns:
            int: The y-coordinate of the center of the window.
        """
        return self._rect.center_y

    @center_y.setter
    def center_y(self, value):
        """
        Setter for the center_y property of the Window instance.

        Args:
            value (int): The new y-coordinate of the center of the window.
        """
        _ = self._rect.center_y
        self._rect.center_y = value

    @property
    def width(self):
        """
        Getter for the width property of the Window instance.

        Returns:
            int: The width of the window.
        """
        return self._rect.width

    @width.setter
    def width(self, value):
        """
        Setter for the width property of the Window instance.

        Args:
            value (int): The new width of the window.
        """
        _ = self._rect.width
        self._rect.width = value

    @property
    def height(self):
        """
        Getter for the height property of the Window instance.

        Returns:
            int: The height of the window.
        """
        return self._rect.height

    @height.setter
    def height(self, value):
        """
        Setter for the height property of the Window instance.

        Args:
            value (int): The new height of the window.
        """
        _ = self._rect.height
        self._rect.height = value

    @property
    def size(self):
        """
        Getter for the size property of the Window instance.

        Returns:
            tuple: The size of the window as a tuple (width, height).
        """
        return self._rect.size

    @size.setter
    def size(self, value):
        """
        Setter for the size property of the Window instance.

        Args:
            value (tuple): The new size of the window as a tuple (width, height).
        """
        _ = self._rect.size
        self._rect.size = value

    @property
    def area(self):
        """
        Getter for the area property of the Window instance.

        Returns:
            int: The area of the window.
        """
        return self._rect.area

    @property
    def box(self):
        """
        Getter for the box property of the Window instance.

        Returns:
            tuple: The box of the window as a tuple (left, top, width, height).
        """
        return self._rect.box

    @box.setter
    def box(self, value):
        """
        Setter for the box property of the Window instance.

        Args:
            value (tuple): The new box of the window as a tuple (left, top, width, height).
        """
        _ = self._rect.box
        self._rect.box = value
