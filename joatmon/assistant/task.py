import threading


class BaseTask:
    def __init__(self, api, **kwargs):
        self.api = api
        self.kwargs = kwargs
        self.event = threading.Event()

    @staticmethod
    def params():
        return ['todo']

    def run(self):
        raise NotImplementedError

    def running(self):
        return not self.event.is_set()

    def start(self):
        self.run()

    def stop(self):
        self.event.set()


def run():
    ...
