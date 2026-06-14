import json
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel

from app.messaging.publisher import publish
from app.models import UrgencyResultMessage, ForecastResultMessage


def make_channel():
    ch = MagicMock()
    ch.default_exchange.publish = AsyncMock()
    return ch


def _routing_key(channel):
    return channel.default_exchange.publish.call_args.kwargs["routing_key"]


def _published_msg(channel):
    return channel.default_exchange.publish.call_args.args[0]


# ── routing ───────────────────────────────────────────────────────────────────

async def test_urgency_result_published_to_correct_queue():
    ch = make_channel()
    await publish(ch, UrgencyResultMessage(deliveryId="d1", urgencyScore=7, aiNote="check"))
    assert _routing_key(ch) == "gridtrack.urgency-results"


async def test_forecast_result_published_to_correct_queue():
    ch = make_channel()
    msg = ForecastResultMessage(
        districtId="mezzeh",
        expectedDeliveries=8,
        staffingRatio=0.75,
        label="Understaffed",
        color="#f59e0b",
        generatedAt="2026-06-14T12:00:00",
    )
    await publish(ch, msg)
    assert _routing_key(ch) == "gridtrack.forecast-results"


# ── message-type header ───────────────────────────────────────────────────────

async def test_urgency_result_has_correct_message_type_header():
    ch = make_channel()
    await publish(ch, UrgencyResultMessage(deliveryId="d1", urgencyScore=5, aiNote="ok"))
    header = _published_msg(ch).headers["message-type"]
    assert header == "gridtrack.application.integrationevents.urgencyresultmessage"


async def test_forecast_result_has_correct_message_type_header():
    ch = make_channel()
    msg = ForecastResultMessage(
        districtId="mezzeh",
        expectedDeliveries=5,
        staffingRatio=1.0,
        label="Optimal",
        color="#22c55e",
        generatedAt="2026-06-14T12:00:00",
    )
    await publish(ch, msg)
    header = _published_msg(ch).headers["message-type"]
    assert header == "gridtrack.application.integrationevents.forecastresultmessage"


# ── body serialization ────────────────────────────────────────────────────────

async def test_body_is_valid_json_with_correct_fields():
    ch = make_channel()
    await publish(ch, UrgencyResultMessage(deliveryId="abc", urgencyScore=9, aiNote="critical"))
    parsed = json.loads(_published_msg(ch).body)
    assert parsed["urgencyScore"] == 9
    assert parsed["aiNote"] == "critical"
    assert parsed["deliveryId"] == "abc"


# ── unknown type ──────────────────────────────────────────────────────────────

async def test_unknown_message_type_skips_publish(caplog):
    class GhostMessage(BaseModel):
        value: str

    ch = make_channel()
    with caplog.at_level(logging.WARNING, logger="app.messaging.publisher"):
        await publish(ch, GhostMessage(value="x"))

    ch.default_exchange.publish.assert_not_awaited()
    assert "No queue mapped" in caplog.text


async def test_unknown_type_does_not_raise():
    class GhostMessage(BaseModel):
        value: str

    ch = make_channel()
    await publish(ch, GhostMessage(value="x"))  # must not raise


async def test_missing_message_type_header_skips_publish(mocker, caplog):
    """Queue mapped but header missing → second warning, no publish."""
    from app.messaging import publisher as pub_mod
    mocker.patch.dict(pub_mod._QUEUE_MAP, {"OrphanMessage": "gridtrack.orphan"})
    # _MESSAGE_TYPE_HEADERS intentionally not patched → key absent

    class OrphanMessage(BaseModel):
        value: str

    ch = make_channel()
    with caplog.at_level(logging.WARNING, logger="app.messaging.publisher"):
        await publish(ch, OrphanMessage(value="x"))

    ch.default_exchange.publish.assert_not_awaited()
    assert "No message-type header mapped" in caplog.text
