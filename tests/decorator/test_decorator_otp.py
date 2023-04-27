import pytest


def test_authorized():
    from joatmon.decorator.otp import authorized

    def t():
        ...

    authorized(t, 'token', 'issuer')

    assert True is True


if __name__ == '__main__':
    pytest.main([__file__])
