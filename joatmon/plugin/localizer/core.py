from joatmon.plugin.core import Plugin


class Localizer(Plugin):
    async def localize(self, language, value):
        raise NotImplementedError
