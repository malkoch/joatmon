class ConsoleReader:
    """
    ConsoleReader class that provides the functionality for reading input from the console.

    Methods:
        read: Reads input from the console.
    """

    def __init__(self):
        """
        Initialize ConsoleReader.
        """
        super(ConsoleReader, self).__init__()

    def read(self):
        """
        Reads input from the console.

        Returns:
            str: The input from the console.
        """
        return input()


class ConsoleWriter:
    """
    ConsoleWriter class that provides the functionality for writing output to the console.

    Methods:
        write: Writes data to the console.
    """

    def __init__(self):
        """
        Initialize ConsoleWriter.
        """
        super(ConsoleWriter, self).__init__()

    def write(self, data):
        """
        Writes data to the console.

        Args:
            data (str): The data to be written to the console.
        """
        print(data)
