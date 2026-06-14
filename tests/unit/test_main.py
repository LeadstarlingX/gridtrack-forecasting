import asyncio
import pytest
import app.main as main_module
from app.main import lifespan, app as fastapi_app


async def test_lifespan_creates_consumer_task_on_startup(mocker):
    stop = asyncio.Event()

    async def mock_consumer():
        await stop.wait()

    mocker.patch("app.main.start_consumer", new=mock_consumer)
    main_module._consumer_task = None

    async with lifespan(fastapi_app):
        assert main_module._consumer_task is not None
        assert not main_module._consumer_task.done()


async def test_lifespan_cancels_consumer_task_on_shutdown(mocker):
    stop = asyncio.Event()

    async def mock_consumer():
        await stop.wait()

    mocker.patch("app.main.start_consumer", new=mock_consumer)

    async with lifespan(fastapi_app):
        task = main_module._consumer_task

    assert task is not None
    assert task.done()


async def test_lifespan_handles_already_cancelled_task(mocker):
    """Shutdown path: await _consumer_task raises CancelledError — must not propagate."""
    async def mock_consumer():
        await asyncio.Event().wait()

    mocker.patch("app.main.start_consumer", new=mock_consumer)

    async with lifespan(fastapi_app):
        pass  # just verify no exception leaks out of the context manager
