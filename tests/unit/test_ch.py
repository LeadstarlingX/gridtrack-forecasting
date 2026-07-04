from unittest.mock import MagicMock, AsyncMock
import app.ch as ch_module
from app.ch import _get_client, ch_query


def test_get_client_creates_connection_once(mocker):
    ch_module._client = None
    mock_conn = MagicMock()
    mock_get = mocker.patch("clickhouse_connect.get_client", return_value=mock_conn)
    _get_client()
    _get_client()
    mock_get.assert_called_once()
    ch_module._client = None


def test_get_client_returns_cached_instance(mocker):
    sentinel = MagicMock()
    ch_module._client = sentinel
    result = _get_client()
    assert result is sentinel
    ch_module._client = None


async def test_ch_query_delegates_to_sync_client(mocker):
    mock_client = MagicMock()
    mock_client.query.return_value = MagicMock(result_rows=[[1, 2]])
    mocker.patch("app.ch._get_client", return_value=mock_client)
    result = await ch_query("SELECT 1")
    mock_client.query.assert_called_once_with("SELECT 1", parameters=None)
    assert result.result_rows == [[1, 2]]


async def test_ch_query_forwards_params(mocker):
    mock_client = MagicMock()
    mock_client.query.return_value = MagicMock(result_rows=[])
    mocker.patch("app.ch._get_client", return_value=mock_client)
    await ch_query("SELECT 1 WHERE x = {v:UInt32}", params={"v": 5})
    mock_client.query.assert_called_once_with(
        "SELECT 1 WHERE x = {v:UInt32}", parameters={"v": 5}
    )
