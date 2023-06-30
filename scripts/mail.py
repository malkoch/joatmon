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
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return ''

    @staticmethod
    def params():
        return []

    def run(self):
        subject = self.kwargs.get('subject', '') or self.api.listen('what do you want the subject to be')
        message = self.kwargs.get('message', '') or self.api.listen('what do you want the content to be')
        receiver = self.kwargs.get('receiver', '') or self.api.listen('what do you want the receiver to be')

        contacts = json.loads(open('iva.json', 'r').read()).get('contacts', [])
        contact = list(filter(lambda x: x['name'] == receiver, contacts))
        if len(contact) > 0:
            receiver = contact[0]['email']

        receivers = [receiver]

        config = json.loads(open('iva.json', 'r').read())['configs']['mail']

        text_subtype = 'plain'
        subject = subject

        smtp_server = config['smtp_server']
        port = config['smtp_port']
        sender_email = config['sender_mail']
        password = config['sender_password']

        msg = MIMEText(message, text_subtype)
        msg['Subject'] = subject
        msg['From'] = sender_email

        conn = smtplib.SMTP(smtp_server, port)
        conn.login(sender_email, password)
        conn.sendmail(sender_email, receivers, msg.as_string())
        conn.quit()

        if not self.event.is_set():
            self.event.set()


class Service(BaseService):
    def __init__(self, api, **kwargs):
        super(Service, self).__init__(api, **kwargs)

        register(KafkaPlugin, 'kafka_plugin', 'localhost:9092')

        self.send_mail = consumer('kafka_plugin', 'mail')(self.send_mail)

    @staticmethod
    def create(api):
        return {}

    def send_mail(self, mail):
        config = json.loads(open('iva.json', 'r').read())['configs']['mail']

        text_subtype = mail['content_type']
        message = mail['message']
        subject = mail['subject']

        smtp_server = config['smtp_server']
        port = config['smtp_port']
        sender_email = config['sender_mail']
        password = config['sender_password']
        receivers = mail['receivers']

        msg = MIMEText(message, text_subtype)
        msg['Subject'] = subject
        msg['From'] = sender_email

        conn = smtplib.SMTP(smtp_server, port)
        conn.login(sender_email, password)
        conn.sendmail(sender_email, receivers, msg.as_string())
        conn.quit()

    def run(self):
        while True:
            if self.event.is_set():
                break
            time.sleep(1)


if __name__ == '__main__':
    Task(None).run()
