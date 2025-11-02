import platform


if platform.system() == "Windows":
    from joatmon.bot.memory._windows import Address
elif platform.system() == "Linux":
    from joatmon.bot.memory._linux import Address
elif platform.system() == "Darwin":
    from joatmon.bot.memory._mac import Address
else:
    raise RuntimeError("Unsupported platform")
