import json
import pickle

from joatmon.core.utility import (
    JSONEncoder,
    to_case,
    to_enumerable
)


class Serializable(object):
    """
    Serializable class for managing serializable objects.

    This class provides a way to manage serializable objects, including their conversion to and from various formats such as dictionary, JSON, and bytes.

    Attributes:
        **kwargs: Variable length keyword arguments.
    """

    def __init__(self, **kwargs):
        """
        Initialize a Serializable object.

        Args:
            **kwargs: Variable length keyword arguments.
        """
        for slot_name, slot_value in kwargs.items():
            if not isinstance(slot_name, str):
                raise ValueError(
                    f'{slot_name} is type {type(slot_name)} is not supported. only string type is supported for field names.'
                )
            setattr(self, slot_name, slot_value)

    def __str__(self):
        """
        Convert the Serializable object to a pretty JSON string.

        Returns:
            str: The pretty JSON string representation of the Serializable object.
        """
        return self.pretty_json

    def __repr__(self):
        """
        Return the string representation of the Serializable object.

        Returns:
            str: The string representation of the Serializable object.
        """
        return str(self)

    def keys(self):
        """
        Yield the keys of the Serializable object.

        Yields:
            str: The keys of the Serializable object.
        """
        for key in self.__dict__.keys():
            yield key

    def values(self):
        """
        Yield the values of the Serializable object.

        Yields:
            Any: The values of the Serializable object.
        """
        for value in self.__dict__.values():
            yield value

    def items(self):
        """
        Yield the key-value pairs of the Serializable object.

        Yields:
            tuple: The key-value pairs of the Serializable object.
        """
        for key, value in self.__dict__.items():
            yield key, value

    def __getitem__(self, key):
        """
        Get the value of the Serializable object for the given key.

        Args:
            key (str): The key to get the value for.

        Returns:
            Any: The value of the Serializable object for the given key.
        """
        return self.__dict__[key]

    @property
    def dict(self):
        """
        Convert the Serializable object to a dictionary.

        Returns:
            dict: The dictionary representation of the Serializable object.
        """
        return to_enumerable(self)

    @classmethod
    def from_dict(cls, dictionary: dict):
        """
        Create a Serializable object from a dictionary.

        Args:
            dictionary (dict): The dictionary to create the Serializable object from.

        Returns:
            Serializable: The Serializable object created from the dictionary.
        """
        return cls(**dictionary)

    @property
    def json(self) -> str:
        """
        Convert the Serializable object to a JSON string.

        Returns:
            str: The JSON string representation of the Serializable object.
        """
        return json.dumps(self.dict, cls=JSONEncoder)

    @property
    def pretty_json(self) -> str:
        """
        Convert the Serializable object to a pretty JSON string.

        Returns:
            str: The pretty JSON string representation of the Serializable object.
        """
        return json.dumps(self.dict, cls=JSONEncoder, indent=4)

    @classmethod
    def from_json(cls, string: str):
        """
        Create a Serializable object from a JSON string.

        Args:
            string (str): The JSON string to create the Serializable object from.

        Returns:
            Serializable: The Serializable object created from the JSON string.
        """
        return cls.from_dict(json.loads(string))

    @property
    def bytes(self) -> bytes:
        """
        Convert the Serializable object to bytes.

        Returns:
            bytes: The bytes representation of the Serializable object.
        """
        return pickle.dumps(self)

    @classmethod
    def from_bytes(cls, value: bytes):
        """
        Create a Serializable object from bytes.

        Args:
            value (bytes): The bytes to create the Serializable object from.

        Returns:
            Serializable: The Serializable object created from the bytes.
        """
        if value is None:
            return None

        obj = pickle.loads(value)
        return obj

    @property
    def snake(self):
        """
        Convert the Serializable object to a dictionary with snake case keys.

        Returns:
            dict: The dictionary representation of the Serializable object with snake case keys.
        """
        return to_case('snake', self.__dict__)

    @property
    def pascal(self):
        """
        Convert the Serializable object to a dictionary with Pascal case keys.

        Returns:
            dict: The dictionary representation of the Serializable object with Pascal case keys.
        """
        return to_case('pascal', self.__dict__)
