from pykafka import KafkaClient
from pykafka.common import OffsetType

from joatmon.plugin.message.core import MessagePlugin


class KafkaPlugin(MessagePlugin):
    def __init__(self, host):
        self.client = KafkaClient(host)

    def get_producer(self, topic):
        return self.client.topics[topic].get_producer()

    def get_consumer(self, topic):
        return self.client.topics[topic].get_simple_consumer(
            auto_commit_enable=True,
            consumer_timeout_ms=1000,
            consumer_group='my_group',
            auto_commit_interval_ms=1000,
            auto_offset_reset=OffsetType.LATEST
        )
