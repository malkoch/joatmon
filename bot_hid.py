import time

from joatmon.bot.hid.mouse import Mouse


if __name__ == "__main__":
    mouse = Mouse()

    mouse.move(x=700, y=1200)
    time.sleep(0.1)
    mouse.click()
    time.sleep(0.5)

    mouse.scroll(dy=500)
