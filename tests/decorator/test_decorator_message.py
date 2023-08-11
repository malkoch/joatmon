import pytest


def test_add_consumer():
    assert True is True


def test_consumer():
    from joatmon.decorator.message import consumer

    def t():
        ...

    consumer('kafka', 'topic')

    assert True is True


def test_consumer_loop_creator():
    assert True is True


def test_loop():
    assert True is True


def test_producer():
    from joatmon.decorator.message import producer

    producer('kafka', 'topic')

    assert True is True


if __name__ == '__main__':
    pytest.main([__file__])
