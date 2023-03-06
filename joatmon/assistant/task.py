import threading


class BaseTask:
    def __init__(self, api=None, background=False, thread_num=1, priority=100, max_run_time=1):
        self.api = api

        self.background = background
        self.thread_num = thread_num
        self.priority = priority
        self.max_run_time = max_run_time
        self.as_job = None
        self.as_service = None

        self.threads = None
        self.event = threading.Event()

        if self.background:
            self.threads = []
            for _ in range(self.thread_num):
                self.threads.append(threading.Thread(target=self.run))

    def __hash__(self):
        return hash(f'{type(self).__module__}.{type(self).__name__}()')

    @classmethod
    def hash(cls):
        return hash(f'{cls.__module__}.{cls.__name__}')

    @staticmethod
    def help(api):
        raise NotImplementedError

    @classmethod
    def create(cls):
        pass

    def run(self):
        # start_time = threading.Timer(self.max_run_time, self.stop)
        # start_time.start()

        if not self.event.is_set():
            self.event.set()

    def running(self):
        return not self.event.is_set()

    def restart(self):
        pass

    def start(self):
        if not self.event.is_set():
            if self.threads is not None:
                for thread in self.threads:
                    thread.start()
            else:
                self.run()

    def stop(self):
        if not self.event.is_set():
            self.event.set()
