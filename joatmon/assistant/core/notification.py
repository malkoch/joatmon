from __future__ import print_function

import time

from joatmon.assistant.service import BaseService
from joatmon.decorator.message import consumer, producer
from joatmon.plugin.core import register
from joatmon.plugin.message.kafka import KafkaPlugin


class Service(BaseService):
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

    def __init__(self, api=None, **kwargs):
        super(Service, self).__init__(api, **kwargs)

        register(KafkaPlugin, 'kafka_plugin', 'localhost:9092')

        self.notification = consumer('kafka_plugin', 'notification')(self.notification)
        self.mail = producer('kafka_plugin', 'mail')(self.mail)

    @staticmethod
    def help():
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return {}

    def notification(self, notification):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        subject = notification['subject']
        content_type = notification['content_type']
        content = notification['content']

        if self.kwargs.get('type', None) == 'mail':
            self.mail(
                {
                    'content_type': content_type,
                    'content': content,
                    'subject': subject,
                    'receivers': self.kwargs.get('receivers', []),
                }
            )

    def mail(self, mail):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def run(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        while True:
            if self.event.is_set():
                break
            time.sleep(1)
