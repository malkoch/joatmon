import os

__all__ = ['Path']


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
