from joatmon.core import context


def register(cls, alias, *args, **kwargs):
    """
    Registers a plugin with a given alias.

    Args:
        cls (str or class): The class or the fully qualified name of the class to be registered.
        alias (str): The alias for the plugin.
        *args: Variable length argument list for the plugin class.
        **kwargs: Arbitrary keyword arguments for the plugin class.
    """
    if isinstance(cls, str):
        try:
            _module = __import__('.'.join(cls.split('.')[:-1]), fromlist=[f'{cls.split(".")[-1]}'])
        except ModuleNotFoundError:
            raise Exception(f'class {cls} is not found')

        cls = getattr(_module, cls.split(".")[-1], None)

    context.set_value(alias.replace('-', '_'), PluginProxy(cls, *args, **kwargs))


class Plugin:
    """
    Abstract Plugin class that defines the interface for plugins.

    This class should be inherited by any class that wants to be used as a plugin.
    """


class PluginProxy:
    """
    PluginProxy class that acts as a proxy for the actual plugin class.

    This class is used to delay the instantiation of the actual plugin class until it is needed.

    Attributes:
        cls (class): The class of the actual plugin.
        args (tuple): The arguments for the plugin class.
        kwargs (dict): The keyword arguments for the plugin class.
        instance (Plugin or None): The instance of the actual plugin class.
    """

    def __init__(self, cls, *args, **kwargs):
        """
        Initialize PluginProxy with the given class, arguments, and keyword arguments.

        Args:
            cls (class): The class of the actual plugin.
            *args: Variable length argument list for the plugin class.
            **kwargs: Arbitrary keyword arguments for the plugin class.
        """
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

        self.instance = None

    def __getattr__(self, item):
        """
        Get an attribute from the actual plugin class.

        This method is called when an attribute lookup has not found the attribute in the usual places.

        Args:
            item (str): The name of the attribute.

        Returns:
            Any: The attribute of the actual plugin class.
        """
        if self.instance is None:
            self.instance = self.cls(*self.args, **self.kwargs)
        return getattr(self.instance, item.replace('-', '_'))

    async def __aenter__(self):
        """
        Define what the context manager should do at the beginning of the block.

        This method is called when the context manager is entered.

        Returns:
            Plugin: The instance of the actual plugin class.
        """
        if self.instance is None:
            self.instance = self.cls(*self.args, **self.kwargs)
        return await self.instance.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Define what the context manager should do after its block has been executed (or terminated).

        This method is called when the context manager is exited.

        Args:
            exc_type (Exception or None): The type of the exception.
            exc_val (Exception or None): The instance of the exception.
            exc_tb (traceback or None): A traceback object encapsulating the call stack at the point where the exception was raised.

        Returns:
            bool: If the method returns a true value, the exception is suppressed.
        """
        if self.instance is None:
            self.instance = self.cls(*self.args, **self.kwargs)
        return await self.instance.__aexit__(exc_type, exc_val, exc_tb)
