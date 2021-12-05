import os
import threading
from contextlib import contextmanager
from threading import Lock

__all__ = ['RWLock', 'Path', 'File', 'Task', 'EventManager']


class RWLock(object):
    def __init__(self):

        self.w_lock = Lock()
        self.num_r_lock = Lock()
        self.num_r = 0

    def r_acquire(self):
        # print('attempting to acquire read lock')
        self.num_r_lock.acquire()
        self.num_r += 1
        if self.num_r == 1:
            self.w_lock.acquire()
        self.num_r_lock.release()

    def r_release(self):
        # print('attempting to release read lock')
        assert self.num_r > 0
        self.num_r_lock.acquire()
        self.num_r -= 1
        if self.num_r == 0:
            self.w_lock.release()
        self.num_r_lock.release()

    @contextmanager
    def r_locked(self):
        try:
            self.r_acquire()
            yield
        finally:
            self.r_release()

    def w_acquire(self):
        # print('attempting to acquire write lock')
        self.w_lock.acquire()

    def w_release(self):
        # print('attempting to release write lock')
        self.w_lock.release()

    @contextmanager
    def w_locked(self):
        try:
            self.w_acquire()
            yield
        finally:
            self.w_release()


class Path:
    def __init__(self, path=''):
        if path is None or path == '':
            path = os.getcwd()
        self.path = path

    def __str__(self):
        return self.path

    def __repr__(self):
        return str(self)

    def __truediv__(self, other):
        if isinstance(other, Path):
            other = other.path

        if other is None or other == '' or other == '/':
            other = '.'

        if other == '.':
            return self

        return Path(os.path.join(self.path, other))

    def __itruediv__(self, other):
        new = self / other
        self.path = new.path

        return self

    def __abs__(self):
        if not self.path.startswith('/'):
            cwd = os.getcwd()
            new = Path(cwd) / self

            return new

        return self

    def exists(self):
        return os.path.exists(self.path)

    def isdir(self):
        return os.path.isdir(self.path)

    def isfile(self):
        return os.path.isfile(self.path)

    def mkdir(self):
        return os.mkdir(self.path)

    def list(self):
        return os.listdir(self.path)


class File:
    def __init__(self, path):
        self.path = path
        self.lock = RWLock()

    def read(self):
        with self.lock.r_locked():
            with open(str(self.path), 'r') as file:
                return file.read()

    def write(self, data=''):
        with self.lock.w_locked():
            with open(str(self.path), 'w') as file:
                file.write(data + '\n')


class Task:
    def __init__(self, iva=None, background=False, thread_num=1, priority=100):
        self.iva = iva

        self.background = background
        self.thread_num = thread_num
        self.priority = priority
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

    @classmethod
    def help(cls):
        return cls.__doc__

    @classmethod
    def create(cls):
        pass

    def run(self):
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


class EventManager:
    def __init__(self):
        self.events = {}

    def add(self, name, order, event):
        if name not in self.events:
            self.events[name] = {}
        if order not in self.events[name]:
            self.events[name][order] = []
        self.events[name][order].append(event)
        return self

    def fire(self, name, order, *args, **kwargs):
        if name not in self.events:
            return
        if order not in self.events[name]:
            return
        for event in self.events[name][order]:
            event(*args, **kwargs)
