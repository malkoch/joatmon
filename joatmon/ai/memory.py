import random


class CoreBuffer(list):
    def __init__(self, values, batch_size):
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
        self.values.append(element)

    def sample(self):
        return random.sample(self, self.batch_size)  # need to implement own random sampling algorithm


class CoreMemory(object):
    """
    Abstract base class for all implemented memory.

    Do not use this abstract base class directly but instead use one of the concrete memory implemented.

    To implement your own memory, you have to implement the following methods:

    - `remember`
    - `sample`
    """

    def __init__(self, buffer, batch_size):
        super(CoreMemory, self).__init__()

        if not isinstance(buffer, CoreBuffer):
            buffer = CoreBuffer([], batch_size)
        else:
            buffer.batch_size = batch_size

        self.buffer = buffer

    def __contains__(self, element):
        return element in self.buffer

    def __getitem__(self, idx):
        return self.buffer[idx]

    def __iter__(self):
        for value in self.buffer:
            yield value

    def __len__(self):
        return len(self.buffer)

    def remember(self, element):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.buffer.add(element)

    def sample(self):
        """
        Sample an experience replay batch with size.

        # Returns
            batch (abstract): Randomly selected batch
            from experience replay memory.
        """
        return self.buffer.sample()


class RingBuffer(CoreBuffer):
    def __init__(self, size, batch_size):
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
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)

        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)


# should have option to, use PER and HER
class RingMemory(CoreMemory):
    """
    Ring Memory

    # Arguments
        size (int): .
    """

    def __init__(self, batch_size=32, size=960000):
        self.buffer = RingBuffer(size=size, batch_size=batch_size)

        super(RingMemory, self).__init__(self.buffer, batch_size)
