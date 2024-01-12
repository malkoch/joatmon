import queue
import threading
import typing

import pika

from joatmon.plugin.message.core import Consumer, MessagePlugin, Producer


class RabbitProducer(Producer):
    def __init__(self, producer):
        self.producer = producer

    def produce(self, topic: str, message: str):
        self.producer.basic_publish(exchange=topic, routing_key=topic, body=message)


class RabbitConsumer(Consumer):
    def __init__(self, consumer):
        self.consumer = consumer

    def set_q(self, q):
        self.q = q

    def consume(self) -> typing.Optional[str]:
        try:
            msg = self.q.get(timeout=.1)
        except queue.Empty:
            return

        return msg.decode('utf-8')


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

    def __init__(self, host, port, username, password):
        self.host = host
        self.port = port
        self.username = username
        self.password = password

        self.q = queue.Queue()

    def get_producer(self, topic):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
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
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        credentials = pika.PlainCredentials(self.username, self.password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port, credentials=credentials))
        channel = connection.channel()

        channel.exchange_declare(topic, durable=True, exchange_type='topic')
        channel.queue_declare(queue=topic)
        channel.queue_bind(exchange=topic, queue=topic, routing_key=topic)

        def callback(ch, method, properties, body):
            self.q.put(body)

        channel.basic_consume(queue=topic, on_message_callback=callback, auto_ack=True)
        threading.Thread(target=channel.start_consuming).start()

        consumer = RabbitConsumer(channel)
        consumer.set_q(self.q)

        return consumer
