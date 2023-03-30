import threading


class BaseService:
    arguments = {

    }

    def __init__(self, api, **kwargs):
        self.api = api
        self.kwargs = kwargs
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.run)

    def run(self):
        if not self.event.is_set():
            self.event.set()

    def running(self):
        return not self.event.is_set()

    def start(self):
        if not self.event.is_set():
            self.thread.start()

    def stop(self):
        if not self.event.is_set():
            self.event.set()
