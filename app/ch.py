"""ClickHouse query helper — sync client wrapped for async use."""

import asyncio
import clickhouse_connect
from app.config import settings

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = clickhouse_connect.get_client(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            database=settings.clickhouse_database,
            username="default",
            connect_timeout=5,
        )
    return _client


async def ch_query(sql: str, params: dict | None = None):
    """Run a ClickHouse query in a thread pool (HTTP client is sync)."""
    client = _get_client()
    return await asyncio.to_thread(client.query, sql, parameters=params)
