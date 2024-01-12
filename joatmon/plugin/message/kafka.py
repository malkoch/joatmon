import typing

import confluent_kafka

from joatmon.plugin.message.core import Consumer, MessagePlugin, Producer


class KafkaProducer(Producer):
    def __init__(self, producer: confluent_kafka.Producer):
        self.producer = producer

    def produce(self, topic: str, message: str):
        self.producer.produce(topic, message)
        self.producer.flush()


class KafkaConsumer(Consumer):
    def __init__(self, consumer: confluent_kafka.Consumer):
        self.consumer = consumer

    def consume(self) -> typing.Optional[str]:
        msg = self.consumer.poll(timeout=1)

        if msg is None:
            return

        if not msg.error():
            self.consumer.commit(asynchronous=False)
            return msg.value()


class KafkaPlugin(MessagePlugin):
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
        self.conf = {
            'bootstrap.servers': host
        }

    def get_producer(self, topic):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.conf['client.id'] = 'joatmon'

        producer = confluent_kafka.Producer(self.conf)

        return KafkaProducer(producer)

    def get_consumer(self, topic):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.conf['group.id'] = 'joatmon'
        self.conf['auto.offset.reset'] = 'earliest'
        self.conf['enable.auto.commit'] = False

        consumer = confluent_kafka.Consumer(self.conf)
        consumer.subscribe([topic])

        return KafkaConsumer(consumer)
