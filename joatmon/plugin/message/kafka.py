import typing

import confluent_kafka

from joatmon.plugin.message.core import Consumer, MessagePlugin, Producer


class KafkaProducer(Producer):
    """
    KafkaProducer class that inherits from the Producer class. It implements the produce method using Kafka.

    Attributes:
        producer (confluent_kafka.Producer): The Kafka producer instance.
    """

    def __init__(self, producer: confluent_kafka.Producer):
        """
        Initialize KafkaProducer with the given Kafka producer.

        Args:
            producer (confluent_kafka.Producer): The Kafka producer instance.
        """
        self.producer = producer

    def produce(self, topic: str, message: str):
        """
        Sends a message to a specified Kafka topic.

        Args:
            topic (str): The topic to which the message should be sent.
            message (str): The message to be sent.
        """
        self.producer.produce(topic, message)
        self.producer.flush()


class KafkaConsumer(Consumer):
    """
    KafkaConsumer class that inherits from the Consumer class. It implements the consume method using Kafka.

    Attributes:
        consumer (confluent_kafka.Consumer): The Kafka consumer instance.
    """

    def __init__(self, consumer: confluent_kafka.Consumer):
        """
        Initialize KafkaConsumer with the given Kafka consumer.

        Args:
            consumer (confluent_kafka.Consumer): The Kafka consumer instance.
        """
        self.consumer = consumer

    def consume(self) -> typing.Optional[str]:
        """
        Receives a message from a Kafka topic.

        Returns:
            str: The received message.
        """
        msg = self.consumer.poll(timeout=1)

        if msg is None:
            return

        if not msg.error():
            self.consumer.commit(asynchronous=False)
            return msg.value()


class KafkaPlugin(MessagePlugin):
    """
    KafkaPlugin class that inherits from the MessagePlugin class. It provides the functionality for producing and consuming messages using Kafka.

    Attributes:
        conf (dict): The configuration for Kafka.
    """

    def __init__(self, host):
        """
        Initialize KafkaPlugin with the given host.

        Args:
            host (str): The host for Kafka.
        """
        self.conf = {
            'bootstrap.servers': host
        }

    def get_producer(self, topic):
        """
        Returns a KafkaProducer for a specified topic.

        Args:
            topic (str): The topic for which a KafkaProducer should be returned.

        Returns:
            KafkaProducer: The KafkaProducer for the specified topic.
        """
        self.conf['client.id'] = 'joatmon'

        producer = confluent_kafka.Producer(self.conf)

        return KafkaProducer(producer)

    def get_consumer(self, topic):
        """
        Returns a KafkaConsumer for a specified topic.

        Args:
            topic (str): The topic for which a KafkaConsumer should be returned.

        Returns:
            KafkaConsumer: The KafkaConsumer for the specified topic.
        """
        self.conf['group.id'] = 'joatmon'
        self.conf['auto.offset.reset'] = 'earliest'
        self.conf['enable.auto.commit'] = False

        consumer = confluent_kafka.Consumer(self.conf)
        consumer.subscribe([topic])

        return KafkaConsumer(consumer)
