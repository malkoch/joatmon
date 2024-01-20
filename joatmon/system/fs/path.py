import os

__all__ = ['Path']


class Path:
    """
    Path class that provides the functionality for handling file paths.

    Attributes:
        path (str): The path of the file.

    Methods:
        __str__: Returns a string representation of the path.
        __repr__: Returns a string representation of the path.
        __truediv__: Joins the current path with another path.
        __add__: Joins the current path with another path.
        __itruediv__: Joins the current path with another path and updates the current path.
        __iadd__: Joins the current path with another path and updates the current path.
        __abs__: Returns the absolute path.
        __dir__: Returns the directory of the path.
        parent: Returns the parent directory of the path.
        exists: Checks if the path exists.
        isdir: Checks if the path is a directory.
        isfile: Checks if the path is a file.
        mkdir: Creates a directory at the path.
        touch: Creates a file at the path.
        list: Lists all files and directories in the path.
    """

    def __init__(self, path=''):
        """
        Initialize Path with the given path.

        Args:
            path (str): The path of the file.
        """
        if path is None or path == '':
            path = os.getcwd()
        self.path = path

    def __str__(self):
        """
        Returns a string representation of the path.

        Returns:
            str: A string representation of the path.
        """
        return self.path

    def __repr__(self):
        """
        Returns a string representation of the path.

        Returns:
            str: A string representation of the path.
        """
        return str(self)

    def __truediv__(self, other):
        """
        Joins the current path with another path.

        Args:
            other (str or Path): The other path.

        Returns:
            Path: The joined path.
        """
        if isinstance(other, Path):
            other = other.path

        if other is None or other == '' or other == '/':
            other = '.'

        if other == '.':
            return self

        return Path(os.path.join(self.path, other))

    def __add__(self, other):
        """
        Joins the current path with another path.

        Args:
            other (str or Path): The other path.

        Returns:
            Path: The joined path.
        """
        return self / other

    def __itruediv__(self, other):
        """
        Joins the current path with another path and updates the current path.

        Args:
            other (str or Path): The other path.

        Returns:
            Path: The updated path.
        """
        new = self / other
        self.path = new.path

        return self

    def __iadd__(self, other):
        """
        Joins the current path with another path and updates the current path.

        Args:
            other (str or Path): The other path.

        Returns:
            Path: The updated path.
        """
        new = self / other
        self.path = new.path

        return self

    def __abs__(self):
        """
        Returns the absolute path.

        Returns:
            Path: The absolute path.
        """
        if not self.path.startswith('/'):
            cwd = os.getcwd()
            new = Path(cwd) / self

            return new

        return self

    def __dir__(self):
        """
        Returns the directory of the path.

        Returns:
            Path: The directory of the path.
        """
        if self.isdir():
            return self
        else:
            return self.parent()

    def parent(self):
        """
        Returns the parent directory of the path.

        Returns:
            Path: The parent directory of the path.
        """
        return Path(os.path.join(self.path, '..'))

    def exists(self):
        """
        Checks if the path exists.

        Returns:
            bool: True if the path exists, False otherwise.
        """
        return os.path.exists(self.path)

    def isdir(self):
        """
        Checks if the path is a directory.

        Returns:
            bool: True if the path is a directory, False otherwise.
        """
        return os.path.isdir(self.path)

    def isfile(self):
        """
        Checks if the path is a file.

        Returns:
            bool: True if the path is a file, False otherwise.
        """
        return os.path.isfile(self.path)

    def mkdir(self):
        """
        Creates a directory at the path.

        Returns:
            None
        """
        return os.mkdir(self.path)

    def touch(self):
        """
        Creates a file at the path.

        Returns:
            None
        """
        return os.mkdir(self.path)

    def list(self):
        """
        Lists all files and directories in the path.

        Returns:
            list: A list of all files and directories in the path.
        """
        return os.listdir(self.path)
