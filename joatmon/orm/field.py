import typing

from joatmon.core.serializable import Serializable


class Field(Serializable):
    def __init__(
            self,
            dtype: typing.Union[type, typing.List, typing.Tuple],
            nullable: bool = True,
            default=None,
            primary: bool = False,
            encrypted: bool = False
    ):
        super(Field, self).__init__()

        self.dtype = dtype
        self.nullable = nullable
        self.primary = primary
        self.encrypted = encrypted

        if not callable(default):
            self.default = lambda: default
        else:
            self.default = default
