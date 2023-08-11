import pytest


def test_log():
    from joatmon.decorator.logger import log

    def t():
        ...

    log('logger')(t)

    assert True is True


if __name__ == '__main__':
    pytest.main([__file__])
