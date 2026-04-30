from joatmon.plugin.cache.core import CachePlugin


class MemoryCache(CachePlugin):
    """
    MemoryCache class that inherits from the CachePlugin class. It implements the abstract methods of the CachePlugin class
    using memory for caching.

    Attributes:
    """

    def __init__(self):
        """
        Initialize MemoryCache

        Args:
        """
        self.data = {}

    async def add(self, key, value, duration=None):
        """
        Add a key-value pair to the cache with an optional duration.

        Args:
            key (str): The key of the item to be added.
            value (str): The value of the item to be added.
            duration (int, optional): The duration for which the item should be stored in the cache.
        """
        self.data[key] = value

    async def get(self, key):
        """
        Retrieve a value from the cache using its key.

        Args:
            key (str): The key of the item to be retrieved.

        Returns:
            str: The value of the item.
        """
        return self.data.get(key)

    async def update(self, key, value):
        """
        Update the value of an item in the cache using its key.

        Args:
            key (str): The key of the item to be updated.
            value (str): The new value of the item.
        """
        self.data[key] = value

    async def remove(self, key):
        """
        Remove an item from the cache using its key.

        Args:
            key (str): The key of the item to be removed.
        """
        keys = list(self.data.keys())
        for k in keys:
            if key not in k:
                continue
            self.data.pop(k)
