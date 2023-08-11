__all__ = ['File']

from joatmon.system.lock import RWLock


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
