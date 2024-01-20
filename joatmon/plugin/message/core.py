from joatmon.plugin.core import Plugin


class Producer:
    """
    Abstract Producer class that defines the interface for producing messages.

    Methods:
        produce: Sends a message to a specified topic.
    """

    def produce(self, topic: str, message: str):
        """
        Sends a message to a specified topic.

        Args:
            topic (str): The topic to which the message should be sent.
            message (str): The message to be sent.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError


class Consumer:
    """
    Abstract Consumer class that defines the interface for consuming messages.

    Methods:
        consume: Receives a message.
    """

    def consume(self) -> str:
        """
        Receives a message.

        Returns:
            str: The received message.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError


class MessagePlugin(Plugin):
    """
    MessagePlugin class that inherits from the Plugin class. It provides the functionality for producing and consuming messages.

    Methods:
        get_producer: Returns a Producer for a specified topic.
        get_consumer: Returns a Consumer for a specified topic.
    """

    def get_producer(self, topic) -> Producer:
        """
        Returns a Producer for a specified topic.

        Args:
            topic (str): The topic for which a Producer should be returned.

        Returns:
            Producer: The Producer for the specified topic.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError

    def get_consumer(self, topic) -> Consumer:
        """
        Returns a Consumer for a specified topic.

        Args:
            topic (str): The topic for which a Consumer should be returned.

        Returns:
            Consumer: The Consumer for the specified topic.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError
