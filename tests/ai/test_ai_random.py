import pytest


def test_core_random_decay():
    assert True is True


def test_core_random_reset():
    assert True is True


def test_core_random_sample():
    assert True is True


def test_gaussian_random_decay():
    from joatmon.ai.random import GaussianRandom

    random = GaussianRandom()
    random.decay()

    assert True is True


def test_gaussian_random_reset():
    from joatmon.ai.random import GaussianRandom

    random = GaussianRandom()
    random.reset()

    assert True is True


def test_gaussian_random_sample():
    from joatmon.ai.random import GaussianRandom

    random = GaussianRandom()
    random.sample()

    assert True is True


def test_ornstein_uhlenbeck_decay():
    from joatmon.ai.random import OrnsteinUhlenbeck

    random = OrnsteinUhlenbeck()
    random.decay()

    assert True is True


def test_ornstein_uhlenbeck_reset():
    from joatmon.ai.random import OrnsteinUhlenbeck

    random = OrnsteinUhlenbeck()
    random.reset()

    assert True is True


def test_ornstein_uhlenbeck_sample():
    from joatmon.ai.random import OrnsteinUhlenbeck

    random = OrnsteinUhlenbeck()
    random.sample()

    assert True is True


if __name__ == '__main__':
    pytest.main([__file__])
