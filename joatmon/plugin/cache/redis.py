import redis

from joatmon.plugin.cache.core import CachePlugin


class RedisCache(CachePlugin):
    """
    RedisCache class that inherits from the CachePlugin class. It implements the abstract methods of the CachePlugin class
    using Redis for caching.

    Attributes:
        host (str): The host of the Redis server.
        port (int): The port of the Redis server.
        password (str): The password for the Redis server.
        connection (`redis.Redis` instance): The connection to the Redis server.
    """

    def __init__(self, host: str, port: int, password: str):
        """
        Initialize RedisCache with the given host, port, and password for the Redis server.

        Args:
            host (str): The host of the Redis server.
            port (int): The port of the Redis server.
            password (str): The password for the Redis server.
        """
        self.host = host
        self.port = port
        self.password = password

        self.connection = redis.Redis(host=host, port=port, password=password)

    async def add(self, key, value, duration=None):
        """
        Add a key-value pair to the cache with an optional duration.

        Args:
            key (str): The key of the item to be added.
            value (str): The value of the item to be added.
            duration (int, optional): The duration for which the item should be stored in the cache.
        """
        self.connection.set(key, value, ex=duration)

    async def get(self, key):
        """
        Retrieve a value from the cache using its key.

        Args:
            key (str): The key of the item to be retrieved.

        Returns:
            str: The value of the item.
        """
        return self.connection.get(key)

    async def update(self, key, value):
        """
        Update the value of an item in the cache using its key.

        Args:
            key (str): The key of the item to be updated.
            value (str): The new value of the item.
        """
        self.connection.set(key, value)

    async def remove(self, key):
        """
        Remove an item from the cache using its key.

        Args:
            key (str): The key of the item to be removed.
        """
        for k in self.connection.keys(key):
            self.connection.delete(k)
