import pytest


def test_authorized():
    from joatmon.decorator.auth import authorized

    def t():
        ...

    authorized(t, 'token', 'issuer')

    assert True is True


if __name__ == '__main__':
    pytest.main([__file__])
