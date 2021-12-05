from joatmon.core import CoreException
from joatmon.plugin.core import Plugin


class MessageQueueException(CoreException):
    ...


class MQ(Plugin):
    def __init__(self, alias: str):
        super(MQ, self).__init__(alias)

    def send(self, topic, message):
        raise NotImplementedError
