from joatmon import context


def register(cls, alias, *args, **kwargs):
    context.set_value(alias.replace('-', '_'), PluginProxy(cls, *args, **kwargs))


class Plugin:
    ...


class PluginProxy:
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

        self.instance = None

    def __getattr__(self, item):
        if self.instance is None:
            self.instance = self.cls(*self.args, **self.kwargs)
        return getattr(self.instance, item.replace('-', '_'))

    async def __aenter__(self):
        if self.instance is None:
            self.instance = self.cls(*self.args, **self.kwargs)
        return await self.instance.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.instance is None:
            self.instance = self.cls(*self.args, **self.kwargs)
        return await self.instance.__aexit__(exc_type, exc_val, exc_tb)
