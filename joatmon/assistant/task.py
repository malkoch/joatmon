import threading


class BaseTask:
    def __init__(self, api, *args, **kwargs):
        self.api = api
        self.args = args
        self.kwargs = kwargs
        self.event = threading.Event()

    @staticmethod
    def params():
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def running(self):
        return not self.event.is_set()

    def start(self):
        self.run()

    def stop(self):
        self.event.set()
