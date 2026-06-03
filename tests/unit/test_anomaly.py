from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.models import DeliveryAnomalyIntegrationEvent
from app.services.anomaly import score_anomaly, URGENCY_BASE, DISTRICT_BOOST


def make_event(anomaly_type="StalePosition", district="mezzeh"):
    return DeliveryAnomalyIntegrationEvent(
        deliveryId=uuid4(),
        districtId=district,
        anomalyType=anomaly_type,
        reason="test reason",
        driverLat=33.5,
        driverLng=36.2,
        occurredAt=datetime.now(timezone.utc),
    )


async def test_score_uses_base_table(mocker):
    mocker.patch("app.services.anomaly._groq_note", return_value="take action")
    result = await score_anomaly(make_event("RouteDeviation", "mezzeh"))
    assert result.urgencyScore == URGENCY_BASE["RouteDeviation"]


async def test_district_boost_is_added(mocker):
    mocker.patch("app.services.anomaly._groq_note", return_value="check driver")
    result = await score_anomaly(make_event("StalePosition", "kafrsousa"))
    expected = min(10, URGENCY_BASE["StalePosition"] + DISTRICT_BOOST["kafrsousa"])
    assert result.urgencyScore == expected


async def test_score_is_capped_at_10(mocker):
    mocker.patch("app.services.anomaly._groq_note", return_value="urgent")
    mocker.patch.dict("app.services.anomaly.URGENCY_BASE", {"StalePosition": 9})
    mocker.patch.dict("app.services.anomaly.DISTRICT_BOOST", {"kafrsousa": 5})
    result = await score_anomaly(make_event("StalePosition", "kafrsousa"))
    assert result.urgencyScore == 10


async def test_groq_failure_uses_fallback_note(mocker):
    mocker.patch("app.services.anomaly._groq_note", side_effect=Exception("Groq down"))
    result = await score_anomaly(make_event("EtaExceeded", "mezzeh"))
    assert isinstance(result.aiNote, str)
    assert len(result.aiNote) > 0


async def test_delivery_id_is_string_in_result(mocker):
    mocker.patch("app.services.anomaly._groq_note", return_value="check")
    event = make_event()
    result = await score_anomaly(event)
    assert result.deliveryId == str(event.deliveryId)
