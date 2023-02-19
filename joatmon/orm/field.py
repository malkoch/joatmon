from joatmon.serializable import Serializable
import typing


class Field(Serializable):
    def __init__(self, kind: typing.Union[type, typing.List, typing.Tuple], nullable=True, fallback=None):
        super(Field, self).__init__()

        self.kind = kind
        self.nullable = nullable
        self.fallback = fallback
