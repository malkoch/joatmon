import pytest


def test_transaction():
    from joatmon.decorator.database import transaction

    def t():
        ...

    transaction(['database'])(t)

    assert True is True


if __name__ == '__main__':
    pytest.main([__file__])
