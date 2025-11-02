import platform


if platform.system() == "Windows":
    from joatmon.bot.hid._windows.mouse import Mouse
elif platform.system() == "Linux":
    from joatmon.bot.hid._linux.mouse import Mouse
elif platform.system() == "Darwin":
    from joatmon.bot.hid._mac.mouse import Mouse
else:
    raise RuntimeError("Unsupported platform")
