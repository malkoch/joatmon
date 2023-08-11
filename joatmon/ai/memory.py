from joatmon.structure.buffer import (
    CoreBuffer,
    RingBuffer
)


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
