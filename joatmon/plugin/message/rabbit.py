from pykafka import KafkaClient
from pykafka.common import OffsetType

from joatmon.plugin.message.core import MessagePlugin


class RabbitMQPlugin(MessagePlugin):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, host):
        self.client = KafkaClient(host)

    def get_producer(self, topic):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self.client.topics[topic].get_producer()

    def get_consumer(self, topic):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self.client.topics[topic].get_simple_consumer(
            auto_commit_enable=True,
            consumer_timeout_ms=1000,
            consumer_group='my_group',
            auto_commit_interval_ms=1000,
            auto_offset_reset=OffsetType.LATEST,
        )
