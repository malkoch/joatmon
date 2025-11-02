import platform


if platform.system() == "Windows":
    from joatmon.bot.hid._windows.keyboard import (
        Keyboard,
        VK
    )
elif platform.system() == "Linux":
    from joatmon.bot.hid._linux.keyboard import (
        Keyboard,
        VK
    )
elif platform.system() == "Darwin":
    from joatmon.bot.hid._mac.keyboard import (
        Keyboard,
        VK
    )
else:
    raise RuntimeError("Unsupported platform")
