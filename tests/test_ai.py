import pytest

from joatmon.ai.memory import RingMemory
from joatmon.ai.policy import (
    EpsilonGreedyPolicy,
    GreedyQPolicy
)
from joatmon.ai.random import (
    GaussianRandom,
    OrnsteinUhlenbeck
)


def test_ring_memory():
    memory = RingMemory(batch_size=10, size=10)
    assert len(memory) == 0
    memory.remember(1)
    assert len(memory) == 1
    memory.remember(2)
    assert len(memory) == 2
    memory.remember(3)
    assert len(memory) == 3
    memory.remember(4)
    assert len(memory) == 4
    memory.remember(5)
    assert len(memory) == 5
    memory.remember(6)
    assert len(memory) == 6
    memory.remember(7)
    assert len(memory) == 7
    memory.remember(8)
    assert len(memory) == 8
    memory.remember(9)
    assert len(memory) == 9
    memory.remember(10)
    assert len(memory) == 10
    memory.remember(11)
    assert len(memory) == 10
    memory.remember(12)
    assert len(memory) == 10
    assert len(memory.sample()) == 10
    assert all([x in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12] for x in memory.sample()])
    assert 3 in memory
    assert 2 not in memory
    assert memory[0] == 3
    for idx, data in enumerate(memory):
        assert data == idx + 3
    with pytest.raises(Exception):
        memory[12]


def test_greedy_q_policy():
    policy = GreedyQPolicy()
    policy.reset()
    policy.decay()
    assert policy.use_network() is True


def test_epsilon_greedy_policy():
    policy = EpsilonGreedyPolicy(max_value=1, min_value=0, decay_steps=2)
    policy.reset()
    assert policy.epsilon == 1
    policy.decay()
    assert policy.epsilon == 0.5
    policy.decay()
    assert policy.epsilon == 0
    assert policy.use_network() is True


def test_gaussian_random():
    policy = GaussianRandom()
    policy.reset()
    policy.decay()
    policy.sample()


def test_ornstein_random():
    policy = OrnsteinUhlenbeck()
    policy.reset()
    policy.decay()
    policy.sample()


if __name__ == '__main__':
    pytest.main([__file__])
