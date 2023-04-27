import pytest


def test_cached():
    from joatmon.decorator.cache import cached

    def t():
        ...

    cached('cache', 5)(t)

    assert True is True


def test_remove():
    from joatmon.decorator.cache import remove

    def t():
        ...

    remove('cache', 'regex')(t)

    assert True is True


if __name__ == '__main__':
    pytest.main([__file__])
