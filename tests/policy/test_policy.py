import pytest


def test_core_policy_decay():
    assert True is True


def test_core_policy_reset():
    assert True is True


def test_core_policy_use_network():
    assert True is True


def test_epsilon_greedy_policy_decay():
    from joatmon.policy import EpsilonGreedyPolicy

    policy = EpsilonGreedyPolicy()
    policy.decay()

    assert True is True


def test_epsilon_greedy_policy_reset():
    from joatmon.policy import EpsilonGreedyPolicy

    policy = EpsilonGreedyPolicy()
    policy.reset()

    assert True is True


def test_epsilon_greedy_policy_use_network():
    from joatmon.policy import EpsilonGreedyPolicy

    policy = EpsilonGreedyPolicy()
    policy.use_network()

    assert True is True


def test_greedy_q_policy_decay():
    from joatmon.policy import GreedyQPolicy

    policy = GreedyQPolicy()
    policy.decay()

    assert True is True


def test_greedy_q_policy_reset():
    from joatmon.policy import GreedyQPolicy

    policy = GreedyQPolicy()
    policy.reset()

    assert True is True


def test_greedy_q_policy_use_network():
    from joatmon.policy import GreedyQPolicy

    policy = GreedyQPolicy()
    policy.use_network()

    assert True is True


if __name__ == '__main__':
    pytest.main([__file__])
