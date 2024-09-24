import os

from joatmon.core.utility import convert_size
from joatmon.system.module import Module


class FileSystemModule(Module):
    def __init__(self, system, root):
        super().__init__(system)

        self._root = root
        self._cwd = '/'

    def _get_host_path(self, path: str):
        if os.path.isabs(path):
            return os.path.abspath(os.path.join(self._root, path))
        else:
            return os.path.abspath(os.path.join(os.path.abspath(self._root), './' + os.path.abspath(os.path.join(self._cwd, path))))

    def _get_system_path(self, path: str):
        if os.path.isabs(path):
            return path.replace(os.path.commonpath((self._root, path)), '', 1)
        else:
            return path

    def stat(self, path: str):
        size, unit = convert_size(os.path.getsize(self._get_host_path(path))).split(' ')
        return {
            'size': size,
            'unit': unit
        }

    def touch(self, file: str):
        with open(self._get_host_path(file), 'w'):
            pass

    def write(self, file: str, content: str):
        with open(self._get_host_path(file), 'w') as f:
            f.write(content)

    def append(self, file: str, content: str):
        with open(self._get_host_path(file), 'a') as f:
            f.write(content)

    def read(self, file: str):
        with open(self._get_host_path(file), 'r') as f:
            return f.read()

    def ls(self, path: str):
        content = list(os.listdir(self._get_host_path(path)))
        content.sort(key=lambda x: (not x.startswith('.'), not os.path.isdir(os.path.join(self._get_host_path(path), x)), x))

        for x in content:
            yield x, self.stat(os.path.join(path, x))

    def cd(self, path: str):  # need to check if path is mapped to host system
        if os.path.isabs(path):
            self._cwd = path
        else:
            self._cwd = os.path.abspath(os.path.join(self._cwd, path))

    def mkdir(self, path: str):
        path = self._get_host_path(path)

        if not os.path.exists(path):
            os.mkdir(path)

    def rm(self, path: str):
        path = self._get_host_path(path)
        if os.path.isdir(path):
            os.rmdir(path)
        elif os.path.isfile(path):
            os.remove(path)

    def exists(self, path: str):
        return os.path.exists(self._get_host_path(path))

    def isdir(self, path: str):
        return os.path.isdir(self._get_host_path(path))

    def isfile(self, path: str):
        return os.path.isfile(self._get_host_path(path))
