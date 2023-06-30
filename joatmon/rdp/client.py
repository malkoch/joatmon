import asyncio
import queue
import time

import numpy as np
import pygame
import pygame.locals

from . import rfb

KEYMAP = {
    'bsp': rfb.KEY_BackSpace,
    'tab': rfb.KEY_Tab,
    'return': rfb.KEY_Return,
    'enter': rfb.KEY_Return,
    'esc': rfb.KEY_Escape,
    'ins': rfb.KEY_Insert,
    'delete': rfb.KEY_Delete,
    'del': rfb.KEY_Delete,
    'home': rfb.KEY_Home,
    'end': rfb.KEY_End,
    'pgup': rfb.KEY_PageUp,
    'pgdn': rfb.KEY_PageDown,
    'left': rfb.KEY_Left,
    'up': rfb.KEY_Up,
    'right': rfb.KEY_Right,
    'down': rfb.KEY_Down,

    'slash': rfb.KEY_BackSlash,
    'bslash': rfb.KEY_BackSlash,
    'fslash': rfb.KEY_ForwardSlash,
    'spacebar': rfb.KEY_SpaceBar,
    'space': rfb.KEY_SpaceBar,
    'sb': rfb.KEY_SpaceBar,

    'f1': rfb.KEY_F1,
    'f2': rfb.KEY_F2,
    'f3': rfb.KEY_F3,
    'f4': rfb.KEY_F4,
    'f5': rfb.KEY_F5,
    'f6': rfb.KEY_F6,
    'f7': rfb.KEY_F7,
    'f8': rfb.KEY_F8,
    'f9': rfb.KEY_F9,
    'f10': rfb.KEY_F10,
    'f11': rfb.KEY_F11,
    'f12': rfb.KEY_F12,
    'f13': rfb.KEY_F13,
    'f14': rfb.KEY_F14,
    'f15': rfb.KEY_F15,
    'f16': rfb.KEY_F16,
    'f17': rfb.KEY_F17,
    'f18': rfb.KEY_F18,
    'f19': rfb.KEY_F19,
    'f20': rfb.KEY_F20,

    'lshift': rfb.KEY_ShiftLeft,
    'shift': rfb.KEY_ShiftLeft,
    'rshift': rfb.KEY_ShiftRight,
    'lctrl': rfb.KEY_ControlLeft,
    'ctrl': rfb.KEY_ControlLeft,
    'rctrl': rfb.KEY_ControlRight,
    'lmeta': rfb.KEY_MetaLeft,
    'meta': rfb.KEY_MetaLeft,
    'rmeta': rfb.KEY_MetaRight,
    'lalt': rfb.KEY_AltLeft,
    'alt': rfb.KEY_AltLeft,
    'ralt': rfb.KEY_AltRight,
    'scrlk': rfb.KEY_Scroll_Lock,
    'sysrq': rfb.KEY_Sys_Req,
    'numlk': rfb.KEY_Num_Lock,
    'caplk': rfb.KEY_Caps_Lock,
    'pause': rfb.KEY_Pause,
    'lsuper': rfb.KEY_Super_L,
    'super': rfb.KEY_Super_L,
    'rsuper': rfb.KEY_Super_R,
    'lhyper': rfb.KEY_Hyper_L,
    'hyper': rfb.KEY_Hyper_L,
    'rhyper': rfb.KEY_Hyper_R,

    'kp0': rfb.KEY_KP_0,
    'kp1': rfb.KEY_KP_1,
    'kp2': rfb.KEY_KP_2,
    'kp3': rfb.KEY_KP_3,
    'kp4': rfb.KEY_KP_4,
    'kp5': rfb.KEY_KP_5,
    'kp6': rfb.KEY_KP_6,
    'kp7': rfb.KEY_KP_7,
    'kp8': rfb.KEY_KP_8,
    'kp9': rfb.KEY_KP_9,
    'kpenter': rfb.KEY_KP_Enter,
}

PYGAME_KEYMAP = {
    pygame.K_BACKSPACE: 'bsp',
    pygame.K_TAB: 'tab',
    pygame.K_RETURN: 'return',
    pygame.K_ESCAPE: 'esc',
    pygame.K_INSERT: 'ins',
    pygame.K_DELETE: 'delete',
    pygame.K_HOME: 'home',
    pygame.K_END: 'end',
    pygame.K_PAGEUP: 'pgup',
    pygame.K_PAGEDOWN: 'pgdn',

    pygame.K_LEFT: 'left',
    pygame.K_UP: 'up',
    pygame.K_RIGHT: 'right',
    pygame.K_DOWN: 'down',
    pygame.K_SLASH: 'slash',
    pygame.K_BACKSLASH: 'bslash',
    pygame.K_SPACE: 'space',

    pygame.K_F1: 'f1',
    pygame.K_F2: 'f2',
    pygame.K_F3: 'f3',
    pygame.K_F4: 'f4',
    pygame.K_F5: 'f5',
    pygame.K_F6: 'f6',
    pygame.K_F7: 'f7',
    pygame.K_F8: 'f8',
    pygame.K_F9: 'f9',
    pygame.K_F10: 'f10',
    pygame.K_F11: 'f11',
    pygame.K_F12: 'f12',
    pygame.K_F13: 'f13',
    pygame.K_F14: 'f14',
    pygame.K_F15: 'f15',

    pygame.K_LSHIFT: 'lshift',
    pygame.K_RSHIFT: 'rshift',
    pygame.K_LCTRL: 'lctrl',
    pygame.K_RCTRL: 'rctrl',
    pygame.K_LMETA: 'lmeta',
    pygame.K_RMETA: 'rmeta',
    pygame.K_LALT: 'lalt',
    pygame.K_RALT: 'ralt',
    pygame.K_SCROLLOCK: 'scrlk',
    pygame.K_SYSREQ: 'sysrq',
    pygame.K_NUMLOCK: 'numlk',
    pygame.K_CAPSLOCK: 'caplk',
    pygame.K_PAUSE: 'pause',
    pygame.K_LSUPER: 'lsuper',
    pygame.K_RSUPER: 'rsuper',
    pygame.K_KP0: 'kp0',
    pygame.K_KP1: 'kp1',
    pygame.K_KP2: 'kp2',
    pygame.K_KP3: 'kp3',
    pygame.K_KP4: 'kp4',
    pygame.K_KP5: 'kp5',
    pygame.K_KP6: 'kp6',
    pygame.K_KP7: 'kp7',
    pygame.K_KP8: 'kp8',
    pygame.K_KP9: 'kp9',
    pygame.K_KP_ENTER: 'kpenter',
}


# KEYMAP = {k: KEYMAP.get(v) for k, v in PYGAME_KEYMAP.items()}


class AuthenticationError(Exception):
    ...


class VNCClient(rfb.RFBClient):
    # RAW_ENCODING = 0
    # COPY_RECTANGLE_ENCODING = 1
    # RRE_ENCODING = 2
    # CORRE_ENCODING = 4
    # HEXTILE_ENCODING = 5
    # ZLIB_ENCODING = 6
    # TIGHT_ENCODING = 7
    # ZLIBHEX_ENCODING = 8
    # ZRLE_ENCODING = 16
    encoding = rfb.ZRLE_ENCODING
    x = 0
    y = 0
    buttons = 0
    screen = None
    image_mode = "RGBX"
    deferred = None

    cursor = None
    cmask = None

    SPECIAL_KEYS_US = "~!@#$%^&*()_+{}|:\"<>?"

    def __init__(self, host, port, loop):
        super(VNCClient, self).__init__(host, port, loop)

        self.force_caps = False
        self.nocursor = False
        self.pseudodesktop = False
        self.pseudocursor = False
        self.shared = True

        self.gui_events = queue.Queue()
        self.mouse_events = queue.Queue()
        self.keyboard_events = queue.Queue()

    def _decode_key(self, key):
        if self.force_caps:
            if key.isupper() or key in self.SPECIAL_KEYS_US:
                key = 'shift-%c' % key

        if len(key) == 1:
            keys = [key]
        else:
            keys = key.split('-')

        keys = [KEYMAP.get(k) or ord(k) for k in keys]

        return keys

    async def _gui_handler(self):
        pygame.init()

        surface = pygame.display.set_mode((self.width, self.height))
        pygame.mouse.set_visible(0)

        while True:
            await asyncio.sleep(0.01, loop=self.loop)

            gui_event = None
            while not self.gui_events.empty():
                gui_event = self.gui_events.get_nowait()
            if gui_event is not None:
                screen, = gui_event
                import cv2
                screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)
                surf = pygame.surfarray.make_surface(np.swapaxes(np.asarray(screen), 0, 1))
                surface.blit(surf, (0, 0))
            pygame.display.update()
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit(1)
                elif event.type == pygame.KEYDOWN:
                    mod = event.mod
                    key = event.key

                    self.keyboard_events.put_nowait(('down', mod, key))
                elif event.type == pygame.KEYUP:
                    mod = event.mod
                    key = event.key

                    self.keyboard_events.put_nowait(('up', mod, key))
                elif event.type == pygame.MOUSEMOTION:
                    x, y = pygame.mouse.get_pos()

                    self.mouse_events.put_nowait(('move', x, y))
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    button = event.button

                    self.mouse_events.put_nowait(('down', button))
                elif event.type == pygame.MOUSEBUTTONUP:
                    button = event.button

                    self.mouse_events.put_nowait(('up', button))

    async def _mouse_handler(self):
        while True:
            await asyncio.sleep(0.01, loop=self.loop)

            if self.mouse_events.empty():
                continue

            mouse_event = self.mouse_events.get_nowait()
            if mouse_event[0] == 'move':
                x, y = mouse_event[1:]
                await self.mouse_move(x, y)
            if mouse_event[0] == 'down':
                button, = mouse_event[1:]
                await self.mouse_down(button)
            if mouse_event[0] == 'up':
                button, = mouse_event[1:]
                await self.mouse_up(button)

    async def _keyboard_handler(self):
        while True:
            await asyncio.sleep(0.01, loop=self.loop)

            if self.keyboard_events.empty():
                continue

            keyboard_event = self.keyboard_events.get_nowait()
            mod, key = keyboard_event[1:]
            if ord('a') <= key <= ord('z'):
                key = chr(key)
            elif ord('A') <= key <= ord('Z'):
                key = chr(key)
            elif key in PYGAME_KEYMAP.keys():
                key = PYGAME_KEYMAP.get(key)
            else:
                continue

            if keyboard_event[0] == 'down':
                await self.key_down(key)
            if keyboard_event[0] == 'up':
                await self.key_up(key)

    async def pause(self, duration):
        await asyncio.sleep(duration, loop=self.loop)

    async def key_press(self, key):
        print('keyPress %s', key)
        await self.key_down(key)
        await self.key_up(key)

    async def key_down(self, key):
        print('keyDown %s', key)
        keys = self._decode_key(key)
        for k in keys:
            await self.key_event(k, down=1)

    async def key_up(self, key):
        print('keyUp %s', key)
        keys = self._decode_key(key)
        for k in keys:
            await self.key_event(k, down=0)

    async def mouse_press(self, button):
        print('mousePress %s', button)
        buttons = self.buttons | (1 << (button - 1))
        await self.mouse_down(button)
        await self.mouse_up(button)

    async def mouse_down(self, button):
        print('mouseDown %s', button)
        self.buttons |= 1 << (button - 1)
        await self.pointer_event(self.x, self.y, buttonmask=self.buttons)

    async def mouse_up(self, button):
        print('mouseUp %s', button)
        self.buttons &= ~(1 << (button - 1))
        await self.pointer_event(self.x, self.y, buttonmask=self.buttons)

    async def capture_screen(self, filename, incremental=0):
        print('captureScreen %s', filename)
        return await self._capture(filename, incremental)

    async def capture_region(self, filename, x, y, w, h, incremental=0):
        print('captureRegion %s', filename)
        return self._capture(filename, incremental, x, y, x + w, y + h)

    async def refresh_screen(self, incremental=0):
        await self.framebuffer_update_request(incremental=incremental)

    async def _capture(self, filename, incremental, *args):
        await self.refresh_screen(incremental)
        await self._capture_save(filename)

    async def _capture_save(self, filename, *args):
        # print('captureSave %s', filename)
        # if args:
        #     capture = self.screen.crop(args)
        # else:
        #     capture = self.screen
        # capture.save(filename)
        ...

    async def expect_screen(self, filename, maxrms=0):
        print('expectScreen %s', filename)
        return self._expect_framebuffer(filename, 0, 0, maxrms)

    async def expect_region(self, filename, x, y, maxrms=0):
        print('expectRegion %s (%s, %s)', filename, x, y)
        return self._expect_framebuffer(filename, x, y, maxrms)

    async def _expect_framebuffer(self, filename, x, y, maxrms):
        # await self.framebuffer_update_request(incremental=1)
        # image = Image.open(filename)
        # w, h = image.size
        # self.expected = image.histogram()
        # await self._expect_compare((x, y, x + w, y + h), maxrms)
        ...

    async def _expect_compare(self, box, maxrms):
        # image = self.screen.crop(box)
        #
        # hist = image.histogram()
        # if len(hist) == len(self.expected):
        #     sum_ = 0
        #     for h, e in zip(hist, self.expected):
        #         sum_ += (h - e) ** 2
        #     rms = math.sqrt(sum_ / len(hist))
        #
        #     print('rms:%s maxrms: %s', int(rms), int(maxrms))
        #     if rms <= maxrms:
        #         return self
        #
        # await self._expect_compare(box, maxrms)
        # await self.framebuffer_update_request(incremental=1)
        ...

    async def mouse_move(self, x, y):
        print('mouseMove %d,%d', x, y)
        self.x, self.y = x, y
        await self.pointer_event(x, y, self.buttons)

    async def mouse_drag(self, x, y, step=1):
        print('mouseDrag %d,%d', x, y)
        if x < self.x:
            xsteps = [self.x - i for i in range(step, self.x - x + 1, step)]
        else:
            xsteps = range(self.x, x, step)

        if y < self.y:
            ysteps = [self.y - i for i in range(step, self.y - y + 1, step)]
        else:
            ysteps = range(self.y, y, step)

        for ypos in ysteps:
            await self.mouse_move(self.x, ypos)
            time.sleep(.2)

        for xpos in xsteps:
            await self.mouse_move(xpos, self.y)
            time.sleep(.2)

        await self.mouse_move(x, y)

    async def set_image_mode(self):
        if self.server_version == 3.889:
            await self.set_pixel_format(bpp=16, depth=16, bigendian=0, truecolor=1, redmax=31, greenmax=63, bluemax=31, redshift=11, greenshift=5, blueshift=0)
            self.image_mode = "BGR;16"
        elif self.truecolor and (not self.bigendian) and self.depth == 24 and self.redmax == 255 and self.greenmax == 255 and self.bluemax == 255:
            pixel = ["X"] * self.bypp
            offsets = [offset // 8 for offset in (self.redshift, self.greenshift, self.blueshift)]
            for offset, color in zip(offsets, "RGB"):
                pixel[offset] = color
            self.image_mode = "".join(pixel)
        else:
            await self.set_pixel_format()

    async def vnc_connection_made(self):
        await self.set_image_mode()
        encodings = [self.encoding]
        if self.pseudocursor or self.nocursor:
            encodings.append(rfb.PSEUDO_CURSOR_ENCODING)
        if self.pseudodesktop:
            encodings.append(rfb.PSEUDO_DESKTOP_SIZE_ENCODING)
        await self.set_encodings(encodings)

        asyncio.ensure_future(self._gui_handler(), loop=self.loop)
        asyncio.ensure_future(self._mouse_handler(), loop=self.loop)
        asyncio.ensure_future(self._keyboard_handler(), loop=self.loop)

    async def vnc_request_password(self):
        if self.password is None:
            await self.writer.close()
            print('password required, but none provided')
            return
        await self.send_password(self.password)

    async def vnc_auth_failed(self, reason):
        print("Cannot connect %s" % reason)

    async def begin_update(self):
        ...

    async def commit_update(self, rectangles=None):
        ...

    async def update_rectangle(self, x, y, width, height, data):
        if not data:
            return

        update = np.frombuffer(data, dtype='uint8').reshape((height, width, 4))
        if self.screen is None:
            self.screen = np.zeros((self.height, self.width, 4), dtype='uint8')
        else:
            self.screen[y: y + update.shape[0], x: x + update.shape[1]] = update

        self.gui_events.put_nowait((self.screen,))

        if not self.rectangles:
            import datetime
            print(datetime.datetime.utcnow().isoformat())
            await self.refresh_screen()

    async def copy_rectangle(self, srcx, srcy, x, y, width, height):
        ...

    async def fill_rectangle(self, x, y, width, height, color):
        await self.update_rectangle(x, y, width, height, color * width * height)

    async def update_cursor(self, x, y, width, height, image, mask):
        if self.nocursor:
            return

        if not width or not height:
            self.cursor = None

        # self.cursor = Image.frombytes('RGBX', (width, height), image)
        # self.cmask = Image.frombytes('1', (width, height), mask)
        self.cfocus = x, y
        await self.draw_cursor()

    async def draw_cursor(self):
        if not self.cursor:
            return

        if not self.screen:
            return

        # x = self.x - self.cfocus[0]
        # y = self.y - self.cfocus[1]
        # self.screen.paste(self.cursor, (x, y), self.cmask)

    async def update_desktop_size(self, width, height):
        # new_screen = Image.new("RGB", (width, height), "black")
        # if self.screen:
        #     new_screen.paste(self.screen, (0, 0))
        # self.screen = new_screen
        ...

    async def bell(self):
        print('ding')

    async def copy_text(self, text):
        print('clipboard copy', repr(text))

    async def paste(self, message):
        await self.client_cut_text(message)
