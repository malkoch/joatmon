import platform


if platform.system() == "Windows":
    from joatmon.bot.memory._windows import Process
elif platform.system() == "Linux":
    from joatmon.bot.memory._linux import Process
elif platform.system() == "Darwin":
    from joatmon.bot.memory._mac import Process
else:
    raise RuntimeError("Unsupported platform")
