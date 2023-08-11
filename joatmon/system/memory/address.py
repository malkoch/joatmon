from joatmon.system.memory.core import hex_dump


class Address(object):
    def __init__(self, value, process, default_type='uint'):
        self.value = int(value)
        self.process = process
        self.default_type = default_type
        self.symbolic_name = None

    def read(self, of_type=None, max_length=None, errors='raise'):
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
        if not of_type:
            of_type = self.default_type
        return self.process.write(self.value, data, of_type=of_type)

    def symbol(self):
        return self.process.get_symbolic_name(self.value)

    def get_instruction(self):
        return self.process.get_instruction(self.value)

    def dump(self, of_type='bytes', size=512, before=32):
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
