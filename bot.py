import time

import cv2
import numpy as np

from joatmon.bot.hid._mac.screen import (
    focus_by_title,
    pid_for_app_name_exact
)
from joatmon.bot.hid.keyboard import (
    Keyboard,
    VK
)
from joatmon.bot.hid.mouse import Mouse
from joatmon.bot.image import (
    draw_box,
    grab_region_np,
    ocr_words,
    resize_image
)


class ScreenInfo:
    def __init__(self, callback):
        self.is_dirty = True
        self.image = None
        self.callback = callback

    def changed(self):
        self.is_dirty = True
        if self.callback is not None and callable(self.callback):
            self.callback()

    def get(self, box=None, debug=False):
        if self.is_dirty or self.image is None or True:
            self.image = grab_region_np(*(box or (0, 0, 2294, 1490)))
            if debug:
                cv2.imshow("debug", self.image)
                cv2.moveWindow("debug", -1600, 0)
                cv2.waitKey(1)
            self.is_dirty = False
            if self.callback is not None and callable(self.callback):
                self.callback()
        return self.image


class TextInfo:
    def __init__(self, screen=None):
        self.screen = screen or ScreenInfo(self.changed)
        self.screen.callback = self.changed
        self.is_dirty = True
        self.texts = None

    def changed(self):
        self.is_dirty = True

    def get(self, box=None, debug=False):
        image = self.screen.get(box, debug)
        if self.is_dirty or self.texts is None or True:
            self.texts = ocr_words(image)
            self.is_dirty = False
        return self.texts


class RaidBot:
    def __init__(self, debug=False):
        self.debug = debug

        self.mouse = Mouse()
        self.keyboard = Keyboard()

        self.screen = ScreenInfo(None)
        self.texts = TextInfo(self.screen)

        self.debug_size = (1366, 720)
        self.screen_size = (2294, 1490)

        if self.debug:
            image = np.zeros((*self.debug_size, 3), dtype=np.uint8)
            cv2.imshow("image", image)
            cv2.moveWindow("image", -1600, 0)

        time.sleep(1)
        cv2.waitKey(100)

        pid = pid_for_app_name_exact('Raid')
        focus_by_title(pid, 'raid')

        time.sleep(1)

        self.go_main_screen()

        self.quests = {}

    def find_energy(self):
        texts = self.texts.get((300, 0, 1100, 200))
        maybe_energy = list(filter(lambda x: '/130' in x['text'].lower(), texts))
        if not maybe_energy:
            return
        maybe_energy = maybe_energy[0]
        maybe_energy = int(''.join(list(filter(str.isdigit, maybe_energy['text'].split('/')[0]))))
        print(f'Energy: {maybe_energy}')

    def close_popup(self):
        self.screen.changed()

    def is_main_screen(self):
        texts = self.texts.get()
        main_screen_texts = {'forge', 'tavern', 'market', 'mine'}

        text_values = [x['text'].lower() for x in texts]

        return all([x in text_values for x in main_screen_texts]) or True

    def go_main_screen(self):
        if self.is_main_screen():
            return

        self.keyboard.press_vkey(VK["ESC"])
        self.screen.changed()
        time.sleep(1)

    def open_market(self):
        self.screen.changed()

    def is_quest_screen(self):
        texts = self.texts.get()
        quest_screen_texts = {'daily', 'weekly', 'monthly', 'advanced'}

        text_values = [x['text'].lower() for x in texts]

        return all([x in text_values for x in quest_screen_texts])

    def open_quests(self):
        texts = self.texts.get(box=(0, 1100, 2294, 390))
        maybe_quests = list(filter(lambda x: 'quests' in x['text'].lower(), texts))
        if maybe_quests:
            maybe_quests = maybe_quests[0]
            maybe_quest_location = maybe_quests['x'] + maybe_quests['w'] // 2, maybe_quests['y'] + 1100 + maybe_quests['h'] // 2
            self.mouse.move(*maybe_quest_location)
            self.mouse.click(*maybe_quest_location)

        self.screen.changed()
        time.sleep(1)

        texts = self.texts.get(box=(50, 200, 500, 800))
        maybe_daily = list(filter(lambda x: 'daily' in x['text'].lower(), texts))
        if maybe_daily:
            maybe_daily = maybe_daily[0]
            maybe_daily_location = maybe_daily['x'] + 50 + maybe_daily['w'] // 2, maybe_daily['y'] + 200 + maybe_daily['h'] // 2
            self.mouse.move(*maybe_daily_location)
            self.mouse.click(*maybe_daily_location)

        self.screen.changed()
        time.sleep(1)

        texts = self.texts.get(box=(500, 500, 700, 800))
        # print(texts)

        self.mouse.move(1100, 1300)
        time.sleep(1)
        self.mouse.scroll(dy=3)

    def retrieve_quests(self):
        self.open_quests()

    def open_shop(self):
        self.screen.changed()

    def open_battle(self):
        self.screen.changed()

    def open_campaign_battle(self):
        self.screen.changed()

    def open_dungeon_battle(self):
        self.screen.changed()

    def open_faction_wars(self):
        self.screen.changed()

    def open_arena(self):
        self.screen.changed()

    def open_clan_boss(self):
        self.screen.changed()

    def open_doom_tower(self):
        self.screen.changed()

    def open_cursed_city(self):
        self.screen.changed()

    def run(self):
        # image = self.screen.get(box=(0, 1100, 2294, 390), debug=True)
        self.retrieve_quests()

        while True:
            image = self.screen.get()
            texts = self.texts.get()

            self.find_energy()
            for text in texts:
                possible_texts = {'shop', 'missions', 'quests', 'challenges', 'clan', 'index', 'champions', 'battle'}

                if text['text'].lower() not in possible_texts:
                    continue
                # if text['text'].lower() == 'battle':
                #     mouse.move(text['x'] + text['w'] // 2, text['y'] + text['h'] // 2)
                #     mouse.click(text['x'] + text['w'] // 2, text['y'] + text['h'] // 2, "left")
                if self.debug:
                    image = draw_box(image, text['x'], text['y'], text['w'], text['h'])

                # if text['text'].lower() == 'quests':
                #     self.open_quests()

            if self.debug:
                image = resize_image(image, *self.debug_size)
                cv2.imshow("image", image)

                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):  # ESC key to break
                    break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    bot = RaidBot(debug=True)
    bot.run()
