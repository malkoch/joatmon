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
        Adds an experience to the memory buffer.

        Args:
            element (tuple): A tuple representing an experience. It includes the state, action, reward, next_state, and terminal flag.
        """
        self.buffer.add(element)

    def sample(self):
        """
        Samples a batch of experiences from the memory buffer.

        Returns:
            list: A list of experiences.
        """
        return self.buffer.sample()


# should have option to, use PER and HER
class RingMemory(CoreMemory):
    """
    Ring Memory

    This class is used to create a ring buffer memory for storing and sampling experiences in reinforcement learning.
    It inherits from the CoreMemory class and overrides its buffer with a RingBuffer.

    Args:
        batch_size (int): The size of the batch to be sampled from the buffer.
        size (int): The maximum size of the ring buffer.
    """

    def __init__(self, batch_size=32, size=960000):
        self.buffer = RingBuffer(size=size, batch_size=batch_size)

        super(RingMemory, self).__init__(self.buffer, batch_size)
