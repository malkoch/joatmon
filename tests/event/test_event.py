import pytest


def test_async_event_fire():
    import asyncio
    from joatmon.event import AsyncEvent

    event = AsyncEvent()

    async def a():
        ...

    event += a

    asyncio.wait_for(asyncio.ensure_future(event.fire()), 10)

    event -= a

    assert True is True


def test_event_fire():
    from joatmon.event import Event

    event = Event()

    def a():
        ...

    event += a

    event.fire()

    event -= a

    assert True is True


if __name__ == '__main__':
    pytest.main([__file__])
