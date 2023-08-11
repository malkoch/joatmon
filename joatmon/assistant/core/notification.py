from __future__ import print_function

import time

from joatmon.assistant.service import BaseService
from joatmon.decorator.message import (
    consumer,
    producer
)
from joatmon.plugin.core import register
from joatmon.plugin.message.kafka import KafkaPlugin


class Service(BaseService):
    def __init__(self, api=None, **kwargs):
        super(Service, self).__init__(api, **kwargs)

        register(KafkaPlugin, 'kafka_plugin', 'localhost:9092')

        self.notification = consumer('kafka_plugin', 'notification')(self.notification)
        self.mail = producer('kafka_plugin', 'mail')(self.mail)

    @staticmethod
    def help():
        return {}

    def notification(self, notification):
        subject = notification['subject']
        content_type = notification['content_type']
        content = notification['content']

        if self.kwargs.get('type', None) == 'mail':
            self.mail(
                {
                    "content_type": content_type,
                    "content": content,
                    "subject": subject,
                    "receivers": self.kwargs.get('receivers', [])
                }
            )

    def mail(self, mail):
        ...

    def run(self):
        while True:
            if self.event.is_set():
                break
            time.sleep(1)
