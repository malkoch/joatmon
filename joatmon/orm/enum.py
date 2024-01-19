import sys
from enum import (
    EnumMeta,
    IntEnum
)

from joatmon.core.utility import (
    to_pascal_string,
    to_snake_string
)


class Meta(EnumMeta):
    """
    Metaclass for Enum class. It extends EnumMeta and overrides the __contains__ method.
    """

    def __contains__(self, item):
        """
        Checks if an item is contained within the Enum class.

        Args:
            item: The item to check.

        Returns:
            bool: True if the item is in the Enum class, False otherwise.
        """
        try:
            self(item)
        except ValueError:
            return False
        else:
            return True


class Enum(IntEnum, metaclass=Meta):
    """
    Enum class that extends IntEnum. It provides additional methods for comparison and string representation.
    """

    def __int__(self):
        """
        Returns the integer value of the Enum member.

        Returns:
            int: The integer value of the Enum member.
        """
        return self.value

    def __str__(self):
        """
        Returns the string representation of the Enum member.

        Returns:
            str: The string representation of the Enum member.
        """
        return f'{to_snake_string(type(self).__name__.replace("Enum", ""))}.{self.name}'
        # return '{@resource.' + f'{to_snake_string(type(self).__name__.replace("Enum", ""))}.{self.name}' + '}'

    def __repr__(self):
        """
        Returns the string representation of the Enum member.

        Returns:
            str: The string representation of the Enum member.
        """
        return f'{type(self).__name__}.{self.name}'

    def __lt__(self, other):
        """
        Checks if the Enum member is less than another.

        Args:
            other: The other value to compare.

        Returns:
            bool: True if the Enum member is less than the other value, False otherwise.

        Raises:
            ValueError: If the other value is not a member of the Enum class.
        """
        if other not in self.__class__:
            raise ValueError(f'given value: {other} could not be casted to {self.__class__}')

        if not isinstance(other, int):
            other = int(other)
        return int(self) < other

    def __gt__(self, other):
        """
        Checks if the Enum member is greater than another.

        Args:
            other: The other value to compare.

        Returns:
            bool: True if the Enum member is greater than the other value, False otherwise.

        Raises:
            ValueError: If the other value is not a member of the Enum class.
        """
        if other not in self.__class__:
            raise ValueError(f'given value: {other} could not be casted to {self.__class__}')

        if not isinstance(other, int):
            other = int(other)
        return int(self) > other

    def __eq__(self, other):
        """
        Checks if the Enum member is equal to another.

        Args:
            other: The other value to compare.

        Returns:
            bool: True if the Enum member is equal to the other value, False otherwise.

        Raises:
            ValueError: If the other value is not a member of the Enum class.
        """
        if other not in self.__class__:
            raise ValueError(f'given value: {other} could not be casted to {self.__class__}')

        if not isinstance(other, int):
            other = int(other)
        return int(self) == other

    @staticmethod
    def parse(value: str):
        """
        Parses a string into an Enum member.

        Args:
            value (str): The string to parse.

        Returns:
            Enum: The parsed Enum member.

        Raises:
            ValueError: If the string cannot be parsed into an Enum member.
        """
        type_name, attribute_name = value.split('.')
        type_name = to_pascal_string(type_name)

        _type = getattr(sys.modules[__name__], f'{type_name}Enum', None)
        if _type is not None:
            return _type[attribute_name]
