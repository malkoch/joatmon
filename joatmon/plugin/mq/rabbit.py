import pika

from joatmon.plugin.mq.core import MQ


class RabbitMQ(MQ):
    def __init__(self, alias: str, connection: str):
        super(RabbitMQ, self).__init__(alias)

        self._channel = pika.BlockingConnection(pika.URLParameters(connection)).channel()

    def send(self, topic, message):
        ...
