import queue
import threading
import typing

import pika

from joatmon.plugin.message.core import Consumer, MessagePlugin, Producer


class RabbitProducer(Producer):
    """
    RabbitProducer class that inherits from the Producer class. It implements the produce method using RabbitMQ.

    Attributes:
        producer (pika.BlockingConnection.channel): The RabbitMQ producer instance.
    """

    def __init__(self, producer):
        """
        Initialize RabbitProducer with the given RabbitMQ producer.

        Args:
            producer (pika.BlockingConnection.channel): The RabbitMQ producer instance.
        """
        self.producer = producer

    def produce(self, topic: str, message: str):
        """
        Sends a message to a specified RabbitMQ topic.

        Args:
            topic (str): The topic to which the message should be sent.
            message (str): The message to be sent.
        """
        self.producer.basic_publish(exchange=topic, routing_key=topic, body=message)


class RabbitConsumer(Consumer):
    """
    RabbitConsumer class that inherits from the Consumer class. It implements the consume method using RabbitMQ.

    Attributes:
        consumer (pika.BlockingConnection.channel): The RabbitMQ consumer instance.
    """

    def __init__(self, consumer):
        """
        Initialize RabbitConsumer with the given RabbitMQ consumer.

        Args:
            consumer (pika.BlockingConnection.channel): The RabbitMQ consumer instance.
        """
        self.consumer = consumer

    def set_q(self, q):
        """
        Set the queue for the consumer.

        Args:
            q (queue.Queue): The queue for the consumer.
        """
        self.q = q

    def consume(self) -> typing.Optional[str]:
        """
        Receives a message from a RabbitMQ topic.

        Returns:
            str: The received message.
        """
        try:
            msg = self.q.get(timeout=.1)
        except queue.Empty:
            return

        return msg.decode('utf-8')


class RabbitMQPlugin(MessagePlugin):
    """
    RabbitMQPlugin class that inherits from the MessagePlugin class. It provides the functionality for producing and consuming messages using RabbitMQ.

    Attributes:
        host (str): The host for RabbitMQ.
        port (int): The port for RabbitMQ.
        username (str): The username for RabbitMQ.
        password (str): The password for RabbitMQ.
        d (dict): A dictionary to store the queues for each topic.
    """

    def __init__(self, host, port, username, password):
        """
        Initialize RabbitMQPlugin with the given host, port, username, and password.

        Args:
            host (str): The host for RabbitMQ.
            port (int): The port for RabbitMQ.
            username (str): The username for RabbitMQ.
            password (str): The password for RabbitMQ.
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password

        self.d = {}

    def get_producer(self, topic):
        """
        Returns a RabbitProducer for a specified topic.

        Args:
            topic (str): The topic for which a RabbitProducer should be returned.

        Returns:
            RabbitProducer: The RabbitProducer for the specified topic.
        """
        if topic not in self.d:
            self.d[topic] = queue.Queue()

        credentials = pika.PlainCredentials(self.username, self.password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port, credentials=credentials))
        channel = connection.channel()

        channel.exchange_declare(topic, durable=True, exchange_type='topic')
        channel.queue_declare(queue=topic)
        channel.queue_bind(exchange=topic, queue=topic, routing_key=topic)

        producer = RabbitProducer(channel)

        return producer

    def get_consumer(self, topic):
        """
        Returns a RabbitConsumer for a specified topic.

        Args:
            topic (str): The topic for which a RabbitConsumer should be returned.

        Returns:
            RabbitConsumer: The RabbitConsumer for the specified topic.
        """
        if topic not in self.d:
            self.d[topic] = queue.Queue()

        credentials = pika.PlainCredentials(self.username, self.password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port, credentials=credentials))
        channel = connection.channel()

        channel.exchange_declare(topic, durable=True, exchange_type='topic')
        channel.queue_declare(queue=topic)
        channel.queue_bind(exchange=topic, queue=topic, routing_key=topic)

        def callback(ch, method, properties, body):
            self.d[topic].put(body)

        channel.basic_consume(queue=topic, on_message_callback=callback, auto_ack=True)
        threading.Thread(target=channel.start_consuming).start()

        consumer = RabbitConsumer(channel)
        consumer.set_q(self.d[topic])

        return consumer
