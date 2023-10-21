from __future__ import print_function

import time

from joatmon.assistant.service import BaseService
from joatmon.assistant.task import BaseTask
from joatmon.decorator.message import consumer


class Task(BaseTask):
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

    def __init__(self, name, api, **kwargs):
        super(Task, self).__init__(name, api, **kwargs)

    @staticmethod
    def help():
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return {
            'name': 'mail',
            'description': 'a function for user to send a mail to someone',
            'parameters': {
                'type': 'object',
                'properties': {
                    'subject': {'type': 'string', 'description': 'subject of the mail'},
                    'message': {'type': 'string', 'description': 'content of the mail'},
                    'receiver': {'type': 'string', 'description': 'receiver of the mail'},
                },
                'required': ['subject', 'message', 'receiver'],
            },
        }

    def run(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """

        if not self.stop_event.is_set():
            self.stop_event.set()


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

    def __init__(self, name, api, **kwargs):
        super(Service, self).__init__(name, api, **kwargs)

        sms_consumer = kwargs.get('consumers', {}).get('sms', {})

        self.send_mail = consumer(sms_consumer.get('plugin'), sms_consumer.get('topic'))(self.send_sms)

    @staticmethod
    def help():
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return {}

    def send_sms(self, sms):
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
