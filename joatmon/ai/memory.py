from joatmon.ai.core import (
    CoreBuffer,
    CoreMemory
)


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
