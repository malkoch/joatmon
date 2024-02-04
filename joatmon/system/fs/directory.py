__all__ = ['File']

from joatmon.core.lock import RWLock


class File:
    """
    File class that provides the functionality for reading and writing to a file with a read-write lock.

    Attributes:
        path (str): The path of the file.
        lock (RWLock): The read-write lock for the file.

    Methods:
        read: Reads the content of the file.
        write: Writes data to the file.
    """

    def __init__(self, path):
        """
        Initialize File with the given path.

        Args:
            path (str): The path of the file.
        """
        self.path = path
        self.lock = RWLock()

    def read(self):
        """
        Reads the content of the file.

        This method uses a read lock to ensure that the file is not modified while it is being read.

        Returns:
            str: The content of the file.
        """
        with self.lock.r_locked():
            with open(str(self.path), 'r') as file:
                return file.read()

    def write(self, data=''):
        """
        Writes data to the file.

        This method uses a write lock to ensure that the file is not read while it is being modified.

        Args:
            data (str): The data to be written to the file.
        """
        with self.lock.w_locked():
            with open(str(self.path), 'w') as file:
                file.write(data + '\n')
