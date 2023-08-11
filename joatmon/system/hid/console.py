class ConsoleReader:
    def __init__(self):
        super(ConsoleReader, self).__init__()

    def read(self):
        return input()


class ConsoleWriter:
    def __init__(self):
        super(ConsoleWriter, self).__init__()

    def write(self, data):
        print(data)
