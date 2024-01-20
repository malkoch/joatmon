import json

from joatmon.core.utility import to_enumerable


def value_to_cls(value, cls):
    """
    Converts a value to a specified class.

    Args:
        value (any): The value to be converted.
        cls (class): The class to convert the value to.

    Returns:
        any: The converted value.
    """
    if isinstance(value, (list, tuple)):
        return [value_to_cls(v, cls) for v in value]
    elif isinstance(value, dict):
        return cls(value)
    else:
        return value


class CustomDictionary(dict):
    """
    CustomDictionary class that inherits from the dict class. It provides the functionality for a dictionary with custom methods.

    Attributes:
        data (dict): The data in the dictionary.

    Methods:
        __str__: Returns a string representation of the dictionary.
        __repr__: Returns a string representation of the dictionary.
        keys: Returns the keys of the dictionary.
        values: Returns the values of the dictionary.
        items: Returns the items of the dictionary.
        __iter__: Returns an iterator over the dictionary.
        __len__: Returns the length of the dictionary.
        __getitem__: Returns the value of a specified key in the dictionary.
        __setitem__: Sets the value of a specified key in the dictionary.
        __getattr__: Returns the value of a specified attribute in the dictionary.
        __setattr__: Sets the value of a specified attribute in the dictionary.
    """

    def __init__(self, data):  # ignore case parameter
        """
        Initialize CustomDictionary with the given data.

        Args:
            data (dict): The data for the dictionary.
        """
        super(CustomDictionary, self).__init__()
        for k, v in data.items():
            self.__dict__[k] = value_to_cls(v, CustomDictionary)

    def __str__(self):
        """
        Returns a string representation of the dictionary.

        Returns:
            str: A string representation of the dictionary.
        """
        return json.dumps(to_enumerable(self.__dict__))

    def __repr__(self):
        """
        Returns a string representation of the dictionary.

        Returns:
            str: A string representation of the dictionary.
        """
        return str(self)

    def keys(self):
        """
        Returns the keys of the dictionary.

        Returns:
            dict_keys: The keys of the dictionary.
        """
        return self.__dict__.keys()

    def values(self):
        """
        Returns the values of the dictionary.

        Returns:
            dict_values: The values of the dictionary.
        """
        return self.__dict__.values()

    def items(self):
        """
        Returns the items of the dictionary.

        Returns:
            dict_items: The items of the dictionary.
        """
        return self.__dict__.items()

    def __iter__(self):
        """
        Returns an iterator over the dictionary.

        Yields:
            tuple: A tuple containing a key-value pair from the dictionary.
        """
        for k, v in self.__dict__.items():
            yield k, v

    def __len__(self):
        """
        Returns the length of the dictionary.

        Returns:
            int: The length of the dictionary.
        """
        return len(self.__dict__)

    def __getitem__(self, item):
        """
        Returns the value of a specified key in the dictionary.

        Args:
            item (str): The key.

        Returns:
            any: The value of the key.
        """
        item_parts = item.split('.')

        curr_item = None
        for idx, item_part in enumerate(item_parts):
            if idx == 0:
                curr_item = self.__dict__.get(item_part, None)
            else:
                if curr_item is None:
                    return curr_item
                else:
                    curr_item = curr_item[item_part]

        return curr_item

    def __setitem__(self, key, value):
        """
        Sets the value of a specified key in the dictionary.

        Args:
            key (str): The key.
            value (any): The value.
        """
        item_parts = key.split('.')

        if len(item_parts) == 1:
            self.__dict__[key] = value
            return

        curr_item = None
        for idx, item_part in enumerate(item_parts[:-1]):
            if idx == 0:
                curr_item = self.__dict__.get(item_part, None)
            else:
                if curr_item is None:
                    return
                else:
                    curr_item = curr_item[item_part]

        curr_item[item_parts[-1]] = value

    def __getattr__(self, item):
        """
        Returns the value of a specified attribute in the dictionary.

        Args:
            item (str): The attribute.

        Returns:
            any: The value of the attribute.
        """
        return self[item]

    def __setattr__(self, key, value):
        """
        Sets the value of a specified attribute in the dictionary.

        Args:
            key (str): The attribute.
            value (any): The value.
        """
        self[key] = value
