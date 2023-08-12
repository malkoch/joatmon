import ctypes
import sys

import cv2
import numpy

__all__ = ['resolution', 'cursor', 'grab']

if sys.platform != 'win32':
    raise Exception('The screen module should only be used on a Windows system.')


class POINT(ctypes.Structure):
    _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]


def resolution():
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    return ctypes.windll.user32.GetSystemMetrics(1), ctypes.windll.user32.GetSystemMetrics(0)


def cursor():
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _cursor = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(_cursor))
    return _cursor.y, _cursor.x


def grab(region=None) -> numpy.ndarray:
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
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
