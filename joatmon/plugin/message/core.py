from joatmon.plugin.core import Plugin


class MessagePlugin(Plugin):
    def get_producer(self, topic):
        raise NotImplementedError

    def get_consumer(self, topic):
        raise NotImplementedError
