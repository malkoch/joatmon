import typing

from joatmon import context
from joatmon.core.serializable import Serializable
from joatmon.orm.enum import Enum


def register(cls, alias, *args, **kwargs):
    context.set_value(alias.replace('-', '_'), PluginProxy(cls, *args, **kwargs))


class Plugin:
    ...


class PluginResponse(Serializable):
    def __init__(
            self,
            data: typing.Optional[typing.Union[dict, list, Serializable]] = None,
            code: Enum = 0
    ):
        super(PluginResponse, self).__init__()
        self.data = data
        self.code = int(code)
        self.message = str(code)


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
