import sys
from enum import EnumMeta, IntEnum

from joatmon.core.utility import to_pascal_string, to_snake_string


class Meta(EnumMeta):
    def __contains__(self, item):
        try:
            self(item)
        except ValueError:
            return False
        else:
            return True


class Enum(IntEnum, metaclass=Meta):
    def __int__(self):
        return self.value

    def __str__(self):
        return f'{to_snake_string(type(self).__name__.replace("Enum", ""))}.{self.name}'
        # return '{@resource.' + f'{to_snake_string(type(self).__name__.replace("Enum", ""))}.{self.name}' + '}'

    def __repr__(self):
        return f'{type(self).__name__}.{self.name}'

    def __lt__(self, other):
        if other not in self.__class__:
            raise ValueError(f'given value: {other} could not be casted to {self.__class__}')

        if not isinstance(other, int):
            other = int(other)
        return int(self) < other

    def __gt__(self, other):
        if other not in self.__class__:
            raise ValueError(f'given value: {other} could not be casted to {self.__class__}')

        if not isinstance(other, int):
            other = int(other)
        return int(self) > other

    def __eq__(self, other):
        if other not in self.__class__:
            raise ValueError(f'given value: {other} could not be casted to {self.__class__}')

        if not isinstance(other, int):
            other = int(other)
        return int(self) == other

    @staticmethod
    def parse(value: str):
        type_name, attribute_name = value.split('.')
        type_name = to_pascal_string(type_name)

        _type = getattr(sys.modules[__name__], f'{type_name}Enum', None)
        if _type is not None:
            return _type[attribute_name]
