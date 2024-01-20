from joatmon.system.memory.core import hex_dump


class Address(object):
    """
    A class used to represent an Address in memory.

    Attributes
    ----------
    value : int
        The value of the address.
    process : object
        The process that the address belongs to.
    default_type : str
        The default type of the address.
    symbolic_name : str
        The symbolic name of the address.

    Methods
    -------
    __init__(self, value, process, default_type='uint')
        Initializes a new instance of the Address class.
    read(self, of_type=None, max_length=None, errors='raise')
        Reads the value at the address.
    write(self, data, of_type=None)
        Writes a value to the address.
    symbol(self)
        Gets the symbolic name of the address.
    get_instruction(self)
        Gets the instruction at the address.
    dump(self, of_type='bytes', size=512, before=32)
        Dumps the memory at the address.
    """

    def __init__(self, value, process, default_type='uint'):
        """
        Initializes a new instance of the Address class.

        Args:
            value (int): The value of the address.
            process (object): The process that the address belongs to.
            default_type (str, optional): The default type of the address. Defaults to 'uint'.
        """
        self.value = int(value)
        self.process = process
        self.default_type = default_type
        self.symbolic_name = None

    def read(self, of_type=None, max_length=None, errors='raise'):
        """
        Reads the value at the address.

        Args:
            of_type (str, optional): The type of the value to read. If not specified, the default type of the address is used.
            max_length (int, optional): The maximum length of the value to read. If not specified, the entire value is read.
            errors (str, optional): The error handling scheme. If 'raise', errors during reading will raise an exception. If 'ignore', errors during reading will be ignored.

        Returns:
            object: The value read from the address.
        """
        if max_length is None:
            try:
                max_length = int(of_type)
                of_type = None
            except Exception as ex:
                print(str(ex))

        if not of_type:
            of_type = self.default_type

        if not max_length:
            return self.process.read(self.value, of_type=of_type, errors=errors)
        else:
            return self.process.read(self.value, of_type=of_type, max_length=max_length, errors=errors)

    def write(self, data, of_type=None):
        """
        Writes a value to the address.

        Args:
            data (object): The value to write to the address.
            of_type (str, optional): The type of the value to write. If not specified, the default type of the address is used.

        Returns:
            int: The number of bytes written.
        """
        if not of_type:
            of_type = self.default_type
        return self.process.write(self.value, data, of_type=of_type)

    def symbol(self):
        """
        Gets the symbolic name of the address.

        Returns:
            str: The symbolic name of the address.
        """
        return self.process.get_symbolic_name(self.value)

    def get_instruction(self):
        """
        Gets the instruction at the address.

        Returns:
            str: The instruction at the address.
        """
        return self.process.get_instruction(self.value)

    def dump(self, of_type='bytes', size=512, before=32):
        """
        Dumps the memory at the address.

        Args:
            of_type (str, optional): The type of the memory to dump. Defaults to 'bytes'.
            size (int, optional): The size of the memory to dump. Defaults to 512.
            before (int, optional): The number of bytes before the address to include in the dump. Defaults to 32.

        Returns:
            None
        """
        buf = self.process.read_bytes(self.value - before, size)
        print(hex_dump(buf, self.value - before, of_type=of_type))

    def __nonzero__(self):
        return self.value is not None and self.value != 0

    def __add__(self, other):
        return Address(self.value + int(other), self.process, self.default_type)

    def __sub__(self, other):
        return Address(self.value - int(other), self.process, self.default_type)

    def __repr__(self):
        if not self.symbolic_name:
            self.symbolic_name = self.symbol()
        return str(f'<Addr: {self.symbolic_name}>')

    def __str__(self):
        if not self.symbolic_name:
            self.symbolic_name = self.symbol()
        return str(f'<Addr: {self.symbolic_name} : "{str(self.read())}" ({self.default_type})>')

    def __int__(self):
        return int(self.value)

    def __hex__(self):
        return hex(self.value)

    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        self.value = int(value)

    def __lt__(self, other):
        return self.value < int(other)

    def __le__(self, other):
        return self.value <= int(other)

    def __eq__(self, other):
        return self.value == int(other)

    def __ne__(self, other):
        return self.value != int(other)

    def __gt__(self, other):
        return self.value > int(other)

    def __ge__(self, other):
        return self.value >= int(other)
