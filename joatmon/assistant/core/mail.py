from __future__ import print_function

import json
import smtplib
import time
from email.mime.text import MIMEText

from joatmon.assistant.service import BaseService
from joatmon.assistant.task import BaseTask
from joatmon.decorator.message import consumer
from joatmon.plugin.core import register
from joatmon.plugin.message.kafka import KafkaPlugin


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

    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

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
        subject = self.kwargs.get('subject', '') or self.api.input('what do you want the subject to be')
        message = self.kwargs.get('message', '') or self.api.input('what do you want the content to be')
        receiver = self.kwargs.get('receiver', '') or self.api.input('what do you want the receiver to be')

        contacts = json.loads(open('iva/iva.json', 'r').read()).get('contacts', [])
        contact = list(filter(lambda x: x['name'] == receiver, contacts))
        if len(contact) > 0:
            receiver = contact[0]['email']

        receivers = [receiver]

        config = json.loads(open('iva/iva.json', 'r').read())['config']['mail']

        text_subtype = 'plain'
        subject = subject

        server = config['server']
        write_port = config['write_port']
        address = config['address']
        password = config['password']

        msg = MIMEText(message, text_subtype)
        msg['Subject'] = subject
        msg['From'] = address

        conn = smtplib.SMTP(server, write_port)
        conn.login(address, password)
        conn.sendmail(address, receivers, msg.as_string())
        conn.quit()

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

    def __init__(self, api, **kwargs):
        super(Service, self).__init__(api, **kwargs)

        register(KafkaPlugin, 'kafka_plugin', 'localhost:9092')

        self.send_mail = consumer('kafka_plugin', 'mail')(self.send_mail)

    @staticmethod
    def help():
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return {}

    def send_mail(self, mail):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        config = json.loads(open('iva/iva.json', 'r').read())['config']['mail']

        text_subtype = mail['content_type']
        content = mail['content']
        subject = mail['subject']

        server = config['server']
        write_port = config['write_port']
        address = config['address']
        password = config['password']
        receivers = mail['receivers']

        msg = MIMEText(content, text_subtype)
        msg['Subject'] = subject
        msg['From'] = address

        conn = smtplib.SMTP(server, write_port)
        conn.login(address, password)
        conn.sendmail(address, receivers, msg.as_string())
        conn.quit()

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
