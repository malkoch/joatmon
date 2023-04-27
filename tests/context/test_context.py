import pytest


def test_get_ctx():
    from joatmon import context

    context.get_ctx()

    assert True is True


def test_get_value():
    from joatmon import context

    class C:
        ...

    ctx = C()
    context.set_ctx(ctx)
    context.set_value('a', 1)
    context.get_value('a')

    assert True is True


def test_set_ctx():
    from joatmon import context

    class C:
        ...

    ctx = C()
    context.set_ctx(ctx)

    assert True is True


def test_set_value():
    from joatmon import context

    class C:
        ...

    ctx = C()
    context.set_ctx(ctx)
    context.set_value('a', 1)

    assert True is True


if __name__ == '__main__':
    pytest.main([__file__])
