from joatmon.core import context
from joatmon.plugin.core import Plugin


class ValuePlugin(Plugin):
    """
    ValuePlugin class that inherits from the Plugin class. It provides the functionality for setting and getting values in the context.

    Attributes:
        key (str): The key for the value in the context.

    Methods:
        set: Sets a value in the context.
        get: Gets a value from the context.
    """

    def __init__(self, key):
        """
        Initialize ValuePlugin with the given key.

        Args:
            key (str): The key for the value in the context.
        """
        self.key = key

    def set(self, value):
        """
        Sets a value in the context.

        This method uses the joatmon.core.context.set_value method to set a value in the context.

        Args:
            value (any): The value to be set in the context.
        """
        context.set_value(self.key, value)

    def get(self):
        """
        Gets a value from the context.

        This method uses the joatmon.core.context.get_value method to get a value from the context.

        Returns:
            any: The value from the context.
        """
        return context.get_value(self.key)
