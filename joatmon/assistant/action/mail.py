from __future__ import print_function

import json
import os
import smtplib
import time
from email.mime.text import MIMEText

from joatmon.assistant.service import BaseService
from joatmon.assistant.task import BaseTask
from joatmon.decorator.message import consumer
from joatmon.system.hid.speaker import play


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
        subject = self.kwargs.get('subject', '') or self.api.input('what do you want the subject to be')
        message = self.kwargs.get('message', '') or self.api.input('what do you want the content to be')
        receiver = self.kwargs.get('receiver', '') or self.api.input('what do you want the receiver to be')

        contacts = json.loads(open(os.path.join(os.environ.get('IVA_PATH'), 'iva.json'), 'r').read()).get('contacts', [])
        contact = list(filter(lambda x: x['name'] == receiver, contacts))
        if len(contact) > 0:
            receiver = contact[0]['email']

        receivers = [receiver]

        config = json.loads(open(os.path.join(os.environ.get('IVA_PATH'), 'iva.json'), 'r').read())['config']['mail']

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

    def __init__(self, name, api, **kwargs):
        super(Service, self).__init__(name, api, **kwargs)

        mail_consumer = kwargs.get('consumers', {}).get('mail', {})

        self.send_mail = consumer(mail_consumer.get('plugin'), mail_consumer.get('topic'))(self.send_mail)

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
        config = json.loads(open(os.path.join(os.environ.get('IVA_PATH'), 'iva.json'), 'r').read())['config']['mail']

        text_subtype = mail['content_type']
        content = mail['content']
        subject = mail['subject']

        server = config['server']
        write_port = config['write_port']
        address = config['address']
        password = config['password']
        to = mail['to']
        cc = mail['cc']
        bcc = mail['bcc']

        contacts = json.loads(open(os.path.join(os.environ.get('IVA_PATH'), 'iva.json'), 'r').read())['contacts']
        mails = set(map(lambda x: x['email'], contacts))
        name_to_mail_mapper = {contact['name']: contact['email'] for contact in contacts}
        alias_to_mail_mapper = {alias: contact['email'] for contact in contacts for alias in contact['aliases']}

        def mail_getter(x):
            if x in mails:
                return x

            if x in name_to_mail_mapper:
                return name_to_mail_mapper[x]

            if x in alias_to_mail_mapper:
                return alias_to_mail_mapper[x]

        to = list(map(lambda x: mail_getter(x), to))
        cc = list(map(lambda x: mail_getter(x), cc))
        bcc = list(map(lambda x: mail_getter(x), bcc))

        to = list(filter(lambda x: x is not None, to))
        cc = list(filter(lambda x: x is not None, cc))
        bcc = list(filter(lambda x: x is not None, bcc))

        msg = MIMEText(content, text_subtype)
        msg['Subject'] = subject
        msg['From'] = address
        msg['To'] = ','.join(to)
        msg['CC'] = ','.join(cc)

        conn = smtplib.SMTP(server, write_port)
        conn.login(address, password)
        conn.sendmail(address, to + cc + bcc, msg.as_string())
        conn.quit()

        play(open(os.path.join(os.path.dirname(__file__), '..', 'assets', 'mail.wav'), 'rb').read())

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
