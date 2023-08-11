import pytest


def test_core_buffer_add():
    from joatmon.ai.memory import CoreBuffer

    buffer = CoreBuffer([], 5)
    buffer.add(1)
    assert True is True


def test_core_buffer_sample():
    from joatmon.ai.memory import CoreBuffer

    buffer = CoreBuffer([], 5)
    buffer.add(1)
    buffer.add(1)
    buffer.add(1)
    buffer.add(1)
    buffer.add(1)
    buffer.sample()

    assert True is True


def test_core_memory_remember():
    from joatmon.ai.memory import CoreMemory, CoreBuffer

    buffer = CoreBuffer([], 5)
    memory = CoreMemory(buffer, 5)
    memory.remember(1)

    assert True is True


def test_core_memory_sample():
    from joatmon.ai.memory import CoreMemory, CoreBuffer

    buffer = CoreBuffer([], 5)
    memory = CoreMemory(buffer, 5)
    memory.remember(1)
    memory.remember(1)
    memory.remember(1)
    memory.remember(1)
    memory.remember(1)
    memory.sample()

    assert True is True


def test_ring_buffer_add():
    from joatmon.ai.memory import RingBuffer

    buffer = RingBuffer(10, 5)
    buffer.add(1)

    assert True is True


def test_ring_buffer_sample():
    from joatmon.ai.memory import RingBuffer

    buffer = RingBuffer(10, 5)
    buffer.add(1)
    buffer.add(1)
    buffer.add(1)
    buffer.add(1)
    buffer.add(1)
    buffer.sample()

    assert True is True


def test_ring_memory_remember():
    from joatmon.ai.memory import RingMemory

    memory = RingMemory(5, 10)
    memory.remember(1)

    assert True is True


def test_ring_memory_sample():
    from joatmon.ai.memory import RingMemory

    memory = RingMemory(5, 10)
    memory.remember(1)
    memory.remember(1)
    memory.remember(1)
    memory.remember(1)
    memory.remember(1)
    memory.sample()

    assert True is True


if __name__ == '__main__':
    pytest.main([__file__])
