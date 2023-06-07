import threading


class BaseJob:
    def __init__(self, api, **kwargs):
        self.api = api
        self.kwargs = kwargs
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.run)

    @staticmethod
    def params():
        return []

    def run(self):
        raise NotImplementedError

    def running(self):
        return not self.event.is_set()

    def start(self):
        self.thread.start()

    def stop(self):
        self.event.set()
