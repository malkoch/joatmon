import platform


if platform.system() == "Windows":
    from joatmon.bot.hid._windows.screen import Screen
elif platform.system() == "Linux":
    from joatmon.bot.hid._linux.screen import Screen
elif platform.system() == "Darwin":
    from joatmon.bot.hid._mac.screen import Screen
else:
    raise RuntimeError("Unsupported platform")
