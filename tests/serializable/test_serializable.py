import pytest


def test_serializable_items():
    from joatmon.serializable import Serializable
    assert list(Serializable(a=12, b=13).items()) == [('a', 12), ('b', 13)]


def test_serializable_keys():
    from joatmon.serializable import Serializable
    assert list(Serializable(a=12, b=13).keys()) == ['a', 'b']


def test_serializable_values():
    from joatmon.serializable import Serializable
    assert list(Serializable(a=12, b=13).values()) == [12, 13]


if __name__ == '__main__':
    pytest.main([__file__])
