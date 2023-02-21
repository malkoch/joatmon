import typing

from joatmon.core.serializable import Serializable


class Field(Serializable):
    def __init__(self, dtype: typing.Union[type, typing.List, typing.Tuple], nullable=True, default=None, primary=False):
        super(Field, self).__init__()

        self.dtype = dtype
        self.nullable = nullable
        self.primary = primary

        if not callable(default):
            self.default = lambda: default
        else:
            self.default = default
