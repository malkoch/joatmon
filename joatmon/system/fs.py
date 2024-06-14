import os

from joatmon.core.utility import convert_size


class FSModule:
    def __init__(self, root):
        self._root = root
        self._cwd = '/'

    def _get_host_path(self, path):
        if os.path.isabs(path):
            return os.path.abspath(os.path.join(self._root, path))
        else:
            return os.path.abspath(os.path.join(os.path.abspath(self._root), './' + os.path.abspath(os.path.join(self._cwd, path))))

    def stat(self, path):
        size, unit = convert_size(os.path.getsize(self._get_host_path(path))).split(' ')
        return {
            'size': size,
            'unit': unit
        }

    def touch(self, file):
        with open(self._get_host_path(file), 'w'):
            pass

    def ls(self, path):
        content = list(os.listdir(self._get_host_path(path)))
        content.sort(key=lambda x: (not x.startswith('.'), not os.path.isdir(os.path.join(self._get_host_path(path), x)), x))

        for x in content:
            yield x, self.stat(os.path.join(path, x))

    def cd(self, path):
        if os.path.isabs(path):
            self._cwd = path
        else:
            self._cwd = os.path.abspath(os.path.join(self._cwd, path))

    def mkdir(self, path):
        path = self._get_host_path(path)

        if not os.path.exists(path):
            os.mkdir(path)

    def rm(self, path):
        path = self._get_host_path(path)
        if os.path.isdir(path):
            os.rmdir(path)
        elif os.path.isfile(path):
            os.remove(path)

    def exists(self, path):
        return os.path.exists(self._get_host_path(path))

    def isdir(self, path):
        return os.path.isdir(self._get_host_path(path))

    def isfile(self, path):
        return os.path.isfile(self._get_host_path(path))
