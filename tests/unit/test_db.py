from unittest.mock import AsyncMock, MagicMock
import app.db as db_module
from app.db import get_pool, close_pool


async def test_get_pool_creates_pool_once(mocker):
    db_module._pool = None
    mock_pool = MagicMock()
    mocker.patch("asyncpg.create_pool", new=AsyncMock(return_value=mock_pool))
    p1 = await get_pool()
    p2 = await get_pool()
    assert p1 is mock_pool
    assert p1 is p2
    db_module._pool = None


async def test_close_pool_closes_and_clears(mocker):
    mock_pool = AsyncMock()
    db_module._pool = mock_pool
    await close_pool()
    mock_pool.close.assert_awaited_once()
    assert db_module._pool is None


async def test_close_pool_is_noop_when_none():
    db_module._pool = None
    await close_pool()  # must not raise
