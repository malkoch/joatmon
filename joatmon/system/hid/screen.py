import ctypes
import sys

import cv2
import numpy

__all__ = ['resolution', 'cursor', 'grab']


class POINT(ctypes.Structure):
    """
    A class used to represent a point with x and y coordinates.

    Attributes
    ----------
    x : ctypes.c_long
        The x-coordinate of the point.
    y : ctypes.c_long
        The y-coordinate of the point.
    """
    _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]


def resolution():
    """
    Gets the resolution of the system.

    Returns:
        tuple: The resolution of the system as a tuple (height, width).
    """
    return ctypes.windll.user32.GetSystemMetrics(1), ctypes.windll.user32.GetSystemMetrics(0)


def cursor():
    """
    Gets the current position of the cursor.

    Returns:
        tuple: The current position of the cursor as a tuple (y, x).
    """
    _cursor = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(_cursor))
    return _cursor.y, _cursor.x


def grab(region=None) -> numpy.ndarray:
    """
    Grabs a screenshot of the specified region.

    Args:
        region (tuple, optional): The region to grab a screenshot of as a tuple (left, top, right, bottom). If not specified, a screenshot of the entire screen is grabbed.

    Returns:
        numpy.ndarray: The screenshot as a numpy array.
    """
    if sys.platform == 'win32':
        import win32api
        import win32con
        import win32gui
        import win32ui

        window_handle = win32gui.GetDesktopWindow()

        if region:
            left, top, right, bot = region
            width = right - left + 1
            height = bot - top + 1
        else:
            # width, height = resolution()
            # left = 0
            # top = 0

            width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
            top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

        window_handle_dc = win32gui.GetWindowDC(window_handle)
        source_dc = win32ui.CreateDCFromHandle(window_handle_dc)
        memory_dc = source_dc.CreateCompatibleDC()
        bit_map = win32ui.CreateBitmap()
        bit_map.CreateCompatibleBitmap(source_dc, width, height)
        memory_dc.SelectObject(bit_map)
        memory_dc.BitBlt((0, 0), (width, height), source_dc, (left, top), win32con.SRCCOPY)

        signed_ints_array = bit_map.GetBitmapBits(True)
        image = numpy.fromstring(signed_ints_array, dtype=numpy.uint8)
        image.shape = (height, width, 4)

        source_dc.DeleteDC()
        memory_dc.DeleteDC()
        win32gui.ReleaseDC(window_handle, window_handle_dc)
        win32gui.DeleteObject(bit_map.GetHandle())

        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
