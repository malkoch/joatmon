import ctypes
import sys

from joatmon.system.decorators import auto_pause

__all__ = ['Keyboard']

if sys.platform != 'win32':
    raise Exception('The keyboard module should only be used on a Windows system.')

SendInput = ctypes.windll.user32.SendInput
MapVirtualKey = ctypes.windll.user32.MapVirtualKeyW

# Constants for failsafe check and pause

FAILSAFE = True
FAILSAFE_POINTS = [(0, 0)]
PAUSE = 0.1  # Tenth-second pause by default.

# Constants for the mouse button names
LEFT = "left"
MIDDLE = "middle"
RIGHT = "right"
PRIMARY = "primary"
SECONDARY = "secondary"

# Mouse Scan Code Mappings
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_LEFTCLICK = MOUSEEVENTF_LEFTDOWN + MOUSEEVENTF_LEFTUP
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_RIGHTCLICK = MOUSEEVENTF_RIGHTDOWN + MOUSEEVENTF_RIGHTUP
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_MIDDLECLICK = MOUSEEVENTF_MIDDLEDOWN + MOUSEEVENTF_MIDDLEUP

# KeyBdInput Flags
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_UNICODE = 0x0004

# MapVirtualKey Map Types
MAPVK_VK_TO_CHAR = 2
MAPVK_VK_TO_VSC = 0
MAPVK_VSC_TO_VK = 1
MAPVK_VSC_TO_VK_EX = 3

PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long),
                ("y", ctypes.c_long)]


class InputI(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", InputI)]


@auto_pause(duration=0.05)
def _send_keyboard_event(key, event):
    if event == 0x00:
        keybd_flags = KEYEVENTF_SCANCODE

        inserted_events = 0
        expected_events = 1

        if key in [0x4B, 0x48, 0x4D, 0x50]:
            keybd_flags |= KEYEVENTF_EXTENDEDKEY
            if ctypes.windll.user32.GetKeyState(0x90):
                expected_events = 2
                hex_key_code = 0xE0
                extra = ctypes.c_ulong(0)
                ii_ = InputI()
                ii_.ki = KeyBdInput(0, hex_key_code, KEYEVENTF_SCANCODE, 0, ctypes.pointer(extra))
                x = Input(ctypes.c_ulong(1), ii_)

                inserted_events += SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

        hex_key_code = key
        extra = ctypes.c_ulong(0)
        ii_ = InputI()
        ii_.ki = KeyBdInput(0, hex_key_code, keybd_flags, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        inserted_events += SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        return inserted_events == expected_events
    elif event == 0x02:
        keybd_flags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP

        inserted_events = 0
        expected_events = 1

        if key in [0x4B, 0x48, 0x4D, 0x50]:
            keybd_flags |= KEYEVENTF_EXTENDEDKEY

        hex_key_code = key
        extra = ctypes.c_ulong(0)
        ii_ = InputI()
        ii_.ki = KeyBdInput(0, hex_key_code, keybd_flags, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)

        inserted_events += SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

        if key in [0x4B, 0x48, 0x4D, 0x50] and ctypes.windll.user32.GetKeyState(0x90):
            expected_events = 2
            hex_key_code = 0xE0
            extra = ctypes.c_ulong(0)
            ii_ = InputI()
            ii_.ki = KeyBdInput(0, hex_key_code, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(1), ii_)
            inserted_events += SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

        return inserted_events == expected_events


class Keyboard:
    KEY_DOWN = 0x0000
    KEY_UP = 0x0002

    VK_BACK = (0x08, 0x0000)
    DK_BACK = (0x0E, 0x0008)

    VK_TAB = (0x09, 0x0000)
    DK_TAB = (0x0F, 0x0008)

    VK_CLEAR = (0x0C, 0x0000)
    DK_CLEAR = (0x4C, 0x0008)

    VK_RETURN = (0x0D, 0x0000)
    DK_RETURN = (0x1C, 0x0008)

    VK_SHIFT = (0x10, 0x0000)
    DK_SHIFT = (0x2A, 0x0008)

    VK_CONTROL = (0x11, 0x0000)
    DK_CONTROL = (0x1D, 0x0008)

    VK_MENU = (0x12, 0x0000)
    DK_MENU = (0x38, 0x0008)

    VK_CAPITAL = (0x14, 0x0000)
    DK_CAPITAL = (0x3A, 0x0008)

    VK_ESCAPE = (0x1B, 0x0000)
    DK_ESCAPE = (0x01, 0x0008)

    VK_SPACE = (0x20, 0x0000)
    DK_SPACE = (0x39, 0x0008)

    VK_PRIOR = (0x21, 0x0000)
    DK_PRIOR = (0x49, 0x0008)

    VK_NEXT = (0x22, 0x0000)
    DK_NEXT = (0x51, 0x0008)

    VK_END = (0x23, 0x0000)
    DK_END = (0x4F, 0x0008)

    VK_HOME = (0x24, 0x0000)
    DK_HOME = (0x47, 0x0008)

    VK_LEFT = (0x25, 0x0000)
    DK_LEFT = (0x4B, 0x0008)

    VK_UP = (0x26, 0x0000)
    DK_UP = (0x48, 0x0008)

    VK_RIGHT = (0x27, 0x0000)
    DK_RIGHT = (0x4D, 0x0008)

    VK_DOWN = (0x28, 0x0000)
    DK_DOWN = (0x50, 0x0008)

    VK_SNAPSHOT = (0x2C, 0x0000)
    DK_SNAPSHOT = (0x54, 0x0008)

    VK_INSERT = (0x2D, 0x0000)
    DK_INSERT = (0x52, 0x0008)

    VK_DELETE = (0x2E, 0x0000)
    DK_DELETE = (0x53, 0x0008)

    VK_HELP = (0x2F, 0x0000)
    DK_HELP = (0x63, 0x0008)

    VK_ALPHA_0 = (0x30, 0x0000)
    DK_ALPHA_0 = (0x0B, 0x0008)

    VK_ALPHA_1 = (0x31, 0x0000)
    DK_ALPHA_1 = (0x02, 0x0008)

    VK_ALPHA_2 = (0x32, 0x0000)
    DK_ALPHA_2 = (0x03, 0x0008)

    VK_ALPHA_3 = (0x33, 0x0000)
    DK_ALPHA_3 = (0x04, 0x0008)

    VK_ALPHA_4 = (0x34, 0x0000)
    DK_ALPHA_4 = (0x05, 0x0008)

    VK_ALPHA_5 = (0x35, 0x0000)
    DK_ALPHA_5 = (0x06, 0x0008)

    VK_ALPHA_6 = (0x36, 0x0000)
    DK_ALPHA_6 = (0x07, 0x0008)

    VK_ALPHA_7 = (0x37, 0x0000)
    DK_ALPHA_7 = (0x08, 0x0008)

    VK_ALPHA_8 = (0x38, 0x0000)
    DK_ALPHA_8 = (0x09, 0x0008)

    VK_ALPHA_9 = (0x39, 0x0000)
    DK_ALPHA_9 = (0x0A, 0x0008)

    VK_A = (0x41, 0x0000)
    DK_A = (0x1E, 0x0008)

    VK_B = (0x42, 0x0000)
    DK_B = (0x30, 0x0008)

    VK_C = (0x43, 0x0000)
    DK_C = (0x2E, 0x0008)

    VK_D = (0x44, 0x0000)
    DK_D = (0x20, 0x0008)

    VK_E = (0x45, 0x0000)
    DK_E = (0x12, 0x0008)

    VK_F = (0x46, 0x0000)
    DK_F = (0x21, 0x0008)

    VK_G = (0x47, 0x0000)
    DK_G = (0x22, 0x0008)

    VK_H = (0x48, 0x0000)
    DK_H = (0x23, 0x0008)

    VK_I = (0x49, 0x0000)
    DK_I = (0x17, 0x0008)

    VK_J = (0x4A, 0x0000)
    DK_J = (0x24, 0x0008)

    VK_K = (0x4B, 0x0000)
    DK_K = (0x25, 0x0008)

    VK_L = (0x4C, 0x0000)
    DK_L = (0x26, 0x0008)

    VK_M = (0x4D, 0x0000)
    DK_M = (0x32, 0x0008)

    VK_N = (0x4E, 0x0000)
    DK_N = (0x31, 0x0008)

    VK_O = (0x4F, 0x0000)
    DK_O = (0x18, 0x0008)

    VK_P = (0x50, 0x0000)
    DK_P = (0x19, 0x0008)

    VK_Q = (0x51, 0x0000)
    DK_Q = (0x10, 0x0008)

    VK_R = (0x52, 0x0000)
    DK_R = (0x13, 0x0008)

    VK_S = (0x53, 0x0000)
    DK_S = (0x1F, 0x0008)

    VK_T = (0x54, 0x0000)
    DK_T = (0x14, 0x0008)

    VK_U = (0x55, 0x0000)
    DK_U = (0x16, 0x0008)

    VK_V = (0x56, 0x0000)
    DK_V = (0x2F, 0x0008)

    VK_W = (0x57, 0x0000)
    DK_W = (0x11, 0x0008)

    VK_X = (0x58, 0x0000)
    DK_X = (0x2D, 0x0008)

    VK_Y = (0x59, 0x0000)
    DK_Y = (0x15, 0x0008)

    VK_Z = (0x5A, 0x0000)
    DK_Z = (0x2C, 0x0008)

    VK_NP_0 = (0x60, 0x0000)
    DK_NP_0 = (0x52, 0x0008)

    VK_NP_1 = (0x61, 0x0000)
    DK_NP_1 = (0x4F, 0x0008)

    VK_NP_2 = (0x62, 0x0000)
    DK_NP_2 = (0x50, 0x0008)

    VK_NP_3 = (0x63, 0x0000)
    DK_NP_3 = (0x51, 0x0008)

    VK_NP_4 = (0x64, 0x0000)
    DK_NP_4 = (0x4B, 0x0008)

    VK_NP_5 = (0x65, 0x0000)
    DK_NP_5 = (0x4C, 0x0008)

    VK_NP_6 = (0x66, 0x0000)
    DK_NP_6 = (0x4D, 0x0008)

    VK_NP_7 = (0x67, 0x0000)
    DK_NP_7 = (0x47, 0x0008)

    VK_NP_8 = (0x68, 0x0000)
    DK_NP_8 = (0x48, 0x0008)

    VK_NP_9 = (0x69, 0x0000)
    DK_NP_9 = (0x49, 0x0008)

    VK_MULTIPLY = (0x6A, 0x0000)
    DK_MULTIPLY = (0x37, 0x0008)

    VK_ADD = (0x6B, 0x0000)
    DK_ADD = (0x4E, 0x0008)

    VK_SUBTRACT = (0x6D, 0x0000)
    DK_SUBTRACT = (0x4A, 0x0008)

    VK_DECIMAL = (0x6E, 0x0000)
    DK_DECIMAL = (0x53, 0x0008)

    VK_F1 = (0x70, 0x0000)
    DK_F1 = (0x3B, 0x0008)

    VK_F2 = (0x71, 0x0000)
    DK_F2 = (0x3C, 0x0008)

    VK_F3 = (0x72, 0x0000)
    DK_F3 = (0x3D, 0x0008)

    VK_F4 = (0x73, 0x0000)
    DK_F4 = (0x3E, 0x0008)

    VK_F5 = (0x74, 0x0000)
    DK_F5 = (0x3F, 0x0008)

    VK_F6 = (0x75, 0x0000)
    DK_F6 = (0x40, 0x0008)

    VK_F7 = (0x76, 0x0000)
    DK_F7 = (0x41, 0x0008)

    VK_F8 = (0x77, 0x0000)
    DK_F8 = (0x42, 0x0008)

    VK_F9 = (0x78, 0x0000)
    DK_F9 = (0x43, 0x0008)

    VK_F10 = (0x79, 0x0000)
    DK_F10 = (0x44, 0x0008)

    VK_F11 = (0x7A, 0x0000)
    DK_F11 = (0x57, 0x0008)

    VK_F12 = (0x7B, 0x0000)
    DK_F12 = (0x58, 0x0008)

    VK_F13 = (0x7C, 0x0000)
    DK_F13 = (0x64, 0x0008)

    VK_F14 = (0x7D, 0x0000)
    DK_F14 = (0x65, 0x0008)

    VK_F15 = (0x7E, 0x0000)
    DK_F15 = (0x66, 0x0008)

    VK_F16 = (0x7F, 0x0000)
    DK_F16 = (0x67, 0x0008)

    VK_F17 = (0x80, 0x0000)
    DK_F17 = (0x68, 0x0008)

    VK_F18 = (0x81, 0x0000)
    DK_F18 = (0x69, 0x0008)

    VK_F19 = (0x82, 0x0000)
    DK_F19 = (0x6A, 0x0008)

    VK_F20 = (0x83, 0x0000)
    DK_F20 = (0x6B, 0x0008)

    VK_F21 = (0x84, 0x0000)
    DK_F21 = (0x6C, 0x0008)

    VK_F22 = (0x85, 0x0000)
    DK_F22 = (0x6D, 0x0008)

    VK_F23 = (0x86, 0x0000)
    DK_F23 = (0x6E, 0x0008)

    VK_F24 = (0x87, 0x0000)
    DK_F24 = (0x76, 0x0008)

    VK_NUMLOCK = (0x90, 0x0000)
    DK_NUMLOCK = (0x45, 0x0008)

    VK_SCROLL = (0x91, 0x0000)
    DK_SCROLL = (0x46, 0x0008)

    VK_LSHIFT = (0xA0, 0x0000)
    DK_LSHIFT = (0x2A, 0x0008)

    VK_RSHIFT = (0xA1, 0x0000)
    DK_RSHIFT = (0x36, 0x0008)

    VK_LCONTROL = (0xA2, 0x0000)
    DK_LCONTROL = (0x1D, 0x0008)

    VK_LMENU = (0xA4, 0x0000)
    DK_LMENU = (0x38, 0x0008)

    VK_OEM_1 = (0xBA, 0x0000)
    DK_OEM_1 = (0x27, 0x0008)

    VK_OEM_PLUS = (0xBB, 0x0000)
    DK_OEM_PLUS = (0x0D, 0x0008)

    VK_OEM_COMMA = (0xBC, 0x0000)
    DK_OEM_COMMA = (0x33, 0x0008)

    VK_OEM_MINUS = (0xBD, 0x0000)
    DK_OEM_MINUS = (0x0C, 0x0008)

    VK_OEM_PERIOD = (0xBE, 0x0000)
    DK_OEM_PERIOD = (0x34, 0x0008)

    VK_OEM_2 = (0xBF, 0x0000)
    DK_OEM_2 = (0x35, 0x0008)

    VK_OEM_3 = (0xC0, 0x0000)
    DK_OEM_3 = (0x29, 0x0008)

    VK_OEM_4 = (0xDB, 0x0000)
    DK_OEM_4 = (0x1A, 0x0008)

    VK_OEM_5 = (0xDC, 0x0000)
    DK_OEM_5 = (0x2B, 0x0008)

    VK_OEM_6 = (0xDD, 0x0000)
    DK_OEM_6 = (0x1B, 0x0008)

    VK_OEM_7 = (0xDE, 0x0000)
    DK_OEM_7 = (0x28, 0x0008)

    VK_OEM_102 = (0xE2, 0x0000)
    DK_OEM_102 = (0x56, 0x0008)

    VK_EREOF = (0xF9, 0x0000)
    DK_EREOF = (0x5D, 0x0008)

    VK_ZOOM = (0xFB, 0x0000)
    DK_ZOOM = (0x62, 0x0008)

    @staticmethod
    def key_down(key):
        key, of_type = key
        _send_keyboard_event(key, Keyboard.KEY_DOWN)

    @staticmethod
    def key_up(key):
        key, of_type = key
        _send_keyboard_event(key, Keyboard.KEY_UP)

    @staticmethod
    def press(key):
        Keyboard.key_down(key)
        Keyboard.key_up(key)
