import random


class CoreBuffer(list):
    """
    CoreBuffer class that inherits from the list class. It provides the functionality for a buffer with core operations.

    Attributes:
        values (list): The list of values in the buffer.
        batch_size (int): The size of the batch for sampling.

    Methods:
        add: Adds an element to the buffer.
        sample: Returns a random sample from the buffer.
    """

    def __init__(self, values, batch_size):
        """
        Initialize CoreBuffer with the given values and batch size.

        Args:
            values (list): The list of values for the buffer.
            batch_size (int): The size of the batch for sampling.
        """
        super(CoreBuffer, self).__init__()

        self.values = values
        self.batch_size = batch_size

    def __contains__(self, element):
        return element in self.values

    def __getitem__(self, idx):
        return self.values[idx]

    def __iter__(self):
        for value in self.values:
            yield value

    def __len__(self):
        return len(self.values)

    def add(self, element):
        """
        Adds an element to the buffer.

        Args:
            element (any): The element to be added to the buffer.
        """
        self.values.append(element)

    def sample(self):
        """
         Returns a random sample from the buffer.

         Returns:
             list: A random sample from the buffer.
         """
        return random.sample(self, self.batch_size)  # need to implement own random sampling algorithm


class RingBuffer(CoreBuffer):
    """
    RingBuffer class that inherits from the CoreBuffer class. It provides the functionality for a circular buffer.

    Attributes:
        data (list): The list of data in the buffer.
        start (int): The start index of the buffer.
        end (int): The end index of the buffer.

    Methods:
        add: Adds an element to the buffer.
    """

    def __init__(self, size, batch_size):
        """
        Initialize RingBuffer with the given size and batch size.

        Args:
            size (int): The size of the buffer.
            batch_size (int): The size of the batch for sampling.
        """
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0

        super(RingBuffer, self).__init__(self.data, batch_size)

    def __contains__(self, item):
        # or return item in self.values
        for value in self:
            if value == item:
                return True

        return False

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise KeyError

        idx = (self.start + idx) % len(self.data)
        return self.data[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        if self.end < self.start:
            return self.end - self.start + len(self.data)
        else:
            return self.end - self.start

    def add(self, element):
        """
        Adds an element to the buffer.

        If the buffer is full, it removes the oldest element to make room for the new element.

        Args:
            element (any): The element to be added to the buffer.
        """
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)

        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
