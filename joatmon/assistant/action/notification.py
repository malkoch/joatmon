from __future__ import print_function

import time

from joatmon.assistant.service import BaseService
from joatmon.decorator.message import (
    consumer,
    producer
)


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

    def __init__(self, name, api=None, **kwargs):
        super(Service, self).__init__(name, api, **kwargs)

        for name, value in kwargs.get('consumers', {}).items():
            plugin = value.get('plugin', None)
            topic = value.get('topic', None)

            self.notification = consumer(plugin, topic)(self.notification)

        mail_producer = kwargs.get('producers', {}).get('mail', {})
        sms_producer = kwargs.get('producers', {}).get('sms', {})

        self.mail = producer(mail_producer.get('plugin'), mail_producer.get('topic'))(self.mail)
        self.sms = producer(sms_producer.get('plugin'), sms_producer.get('topic'))(self.sms)

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
        content = notification['content']
        notification_type = notification['type']

        if notification_type == 'mail':
            self.mail(
                {
                    'content_type': notification['content_type'],
                    'content': content,
                    'subject': notification['subject'],
                    'to': notification.get('to', []),
                    'cc': notification.get('cc', []),
                    'bcc': notification.get('bcc', [])
                }
            )
        if notification_type == 'sms':
            self.sms(
                {
                    'content': content,
                    'to': notification.get('to', []),
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

    def sms(self, sms):
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
