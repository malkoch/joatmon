import typing

from joatmon.serializable import Serializable


class Field(Serializable):
    def __init__(
            self,
            dtype: typing.Union[type, typing.List, typing.Tuple],
            nullable: bool = True,
            default=None,
            primary: bool = False,
            encrypt: bool = False,
            hash_: bool = False
    ):
        super(Field, self).__init__()

        self.dtype = dtype
        self.nullable = nullable
        self.primary = primary
        self.encrypt = encrypt
        self.hash_ = hash_

        if not callable(default):
            self.default = lambda: default
        else:
            self.default = default

        self.encrypted = False
        self.hashed = False
