import pytest


def test_get():
    from joatmon.decorator.web import get

    def t():
        ...

    get(t)

    assert True is True


def test_incoming():
    from joatmon.decorator.web import incoming

    def t():
        ...

    incoming('case', 'json', 'arg', 'form')(t)

    assert True is True


def test_ip_limit():
    from joatmon.decorator.web import ip_limit

    def t():
        ...

    ip_limit(10, 'cache', 'ip')(t)

    assert True is True


def test_limit():
    from joatmon.decorator.web import limit

    def t():
        ...

    limit(10, 'cache')(t)

    assert True is True


def test_outgoing():
    from joatmon.decorator.web import outgoing

    def t():
        ...

    outgoing('case')(t)

    assert True is True


def test_post():
    from joatmon.decorator.web import post

    def t():
        ...

    post(t)

    assert True is True


def test_wrap():
    from joatmon.decorator.web import wrap

    def t():
        ...

    wrap(t)

    assert True is True


if __name__ == '__main__':
    pytest.main([__file__])
