"""
Cross-service contract tests.

These tests verify that the JSON schema and AMQP header values Python produces
exactly match what the .NET Wolverine handlers expect. A failure here means
the E2E pipeline would silently break even when unit and integration tests pass.

No infrastructure required — all tests run offline.
"""
import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from app.models import (
    DeliveryAnomalyIntegrationEvent,
    DriverPositionIntegrationEvent,
    ForecastResultMessage,
    UrgencyResultMessage,
)
from app.messaging.publisher import _MESSAGE_TYPE_HEADERS, _QUEUE_MAP


# ── Outbound message-type header contract ─────────────────────────────────────
# Wolverine (WolverineFx 5.x) identifies incoming message types by reading the
# AMQP `message-type` header.  For interoperability Wolverine uses the full
# lowercased .NET type name: "{namespace}.{classname}".
# The namespace is GridTrack.Application.IntegrationEvents (from C# project).

EXPECTED_URGENCY_TYPE  = "gridtrack.application.integrationevents.urgencyresultmessage"
EXPECTED_FORECAST_TYPE = "gridtrack.application.integrationevents.forecastresultmessage"

EXPECTED_URGENCY_QUEUE  = "gridtrack.urgency-results"
EXPECTED_FORECAST_QUEUE = "gridtrack.forecast-results"


def test_urgency_message_type_header_matches_wolverine_format():
    """The header value Python sends must equal Wolverine's computed type name."""
    assert _MESSAGE_TYPE_HEADERS["UrgencyResultMessage"] == EXPECTED_URGENCY_TYPE


def test_forecast_message_type_header_matches_wolverine_format():
    assert _MESSAGE_TYPE_HEADERS["ForecastResultMessage"] == EXPECTED_FORECAST_TYPE


def test_urgency_queue_name_matches_dotnet_listener():
    """Queue name must match opts.ListenToRabbitQueue("...") in Program.cs."""
    assert _QUEUE_MAP["UrgencyResultMessage"] == EXPECTED_URGENCY_QUEUE


def test_forecast_queue_name_matches_dotnet_listener():
    assert _QUEUE_MAP["ForecastResultMessage"] == EXPECTED_FORECAST_QUEUE


def test_no_extra_or_missing_message_type_entries():
    """Exactly two outbound message types — adding one without wiring .NET would silently drop it."""
    assert set(_MESSAGE_TYPE_HEADERS.keys()) == {"UrgencyResultMessage", "ForecastResultMessage"}
    assert set(_QUEUE_MAP.keys()) == {"UrgencyResultMessage", "ForecastResultMessage"}


# ── UrgencyResultMessage JSON schema ─────────────────────────────────────────
# C# record: UrgencyResultMessage(Guid DeliveryId, int UrgencyScore, string AiNote)
# Wolverine deserializes JSON using System.Text.Json with camelCase naming policy.

def test_urgency_result_json_field_names_are_camelcase():
    """Field names must be camelCase — C# PascalCase properties with Wolverine's camelCase policy."""
    msg = UrgencyResultMessage(
        deliveryId=str(uuid4()),
        urgencyScore=5,
        aiNote="check driver",
    )
    data = json.loads(msg.model_dump_json())
    assert set(data.keys()) == {"deliveryId", "urgencyScore", "aiNote"}


def test_urgency_result_delivery_id_is_hyphenated_uuid_string():
    """C# Guid is deserialized from a hyphenated UUID string (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)."""
    fixed_id = UUID("12345678-1234-5678-1234-567812345678")
    msg = UrgencyResultMessage(
        deliveryId=str(fixed_id),
        urgencyScore=7,
        aiNote="act now",
    )
    data = json.loads(msg.model_dump_json())
    assert data["deliveryId"] == "12345678-1234-5678-1234-567812345678"


def test_urgency_result_score_is_integer():
    msg = UrgencyResultMessage(deliveryId=str(uuid4()), urgencyScore=3, aiNote="ok")
    data = json.loads(msg.model_dump_json())
    assert isinstance(data["urgencyScore"], int)


def test_urgency_result_ai_note_is_string():
    msg = UrgencyResultMessage(deliveryId=str(uuid4()), urgencyScore=4, aiNote="some note")
    data = json.loads(msg.model_dump_json())
    assert isinstance(data["aiNote"], str)


# ── ForecastResultMessage JSON schema ─────────────────────────────────────────
# C# record: ForecastResultMessage(string DistrictId, int ExpectedDeliveries,
#             double StaffingRatio, string Label, string Color, DateTime GeneratedAt)

def test_forecast_result_json_field_names_are_camelcase():
    """All six fields must be present as camelCase."""
    msg = ForecastResultMessage(
        districtId="mezzeh",
        expectedDeliveries=10,
        staffingRatio=0.75,
        label="Moderate",
        color="#fbbf24",
        generatedAt=datetime.now(timezone.utc).isoformat(),
    )
    data = json.loads(msg.model_dump_json())
    assert set(data.keys()) == {
        "districtId", "expectedDeliveries", "staffingRatio",
        "label", "color", "generatedAt",
    }


def test_forecast_result_expected_deliveries_is_integer():
    msg = ForecastResultMessage(
        districtId="babtouma", expectedDeliveries=8,
        staffingRatio=0.6, label="Critical", color="#f87171",
        generatedAt=datetime.now(timezone.utc).isoformat(),
    )
    data = json.loads(msg.model_dump_json())
    assert isinstance(data["expectedDeliveries"], int)


def test_forecast_result_staffing_ratio_is_float():
    msg = ForecastResultMessage(
        districtId="malki", expectedDeliveries=5,
        staffingRatio=0.85, label="Low demand", color="#34d399",
        generatedAt=datetime.now(timezone.utc).isoformat(),
    )
    data = json.loads(msg.model_dump_json())
    assert isinstance(data["staffingRatio"], float)


def test_forecast_result_generated_at_is_parseable_iso8601_string():
    """C# DateTime.Parse() must be able to read this string."""
    now = datetime.now(timezone.utc)
    msg = ForecastResultMessage(
        districtId="mezzeh", expectedDeliveries=2, staffingRatio=1.0,
        label="Low demand", color="#34d399", generatedAt=now.isoformat(),
    )
    data = json.loads(msg.model_dump_json())
    # Must be parseable — C# DateTime.Parse handles ISO-8601 with timezone offset
    parsed = datetime.fromisoformat(data["generatedAt"])
    assert parsed is not None


def test_forecast_result_label_and_color_are_valid_enum_values():
    """C# code switches on these strings — any new value is a silent no-op."""
    valid_labels = {"Critical", "Moderate", "Low demand"}
    valid_colors = {"#f87171", "#fbbf24", "#34d399"}
    for label, color in [("Critical", "#f87171"), ("Moderate", "#fbbf24"), ("Low demand", "#34d399")]:
        msg = ForecastResultMessage(
            districtId="mezzeh", expectedDeliveries=5, staffingRatio=0.5,
            label=label, color=color, generatedAt=datetime.now(timezone.utc).isoformat(),
        )
        data = json.loads(msg.model_dump_json())
        assert data["label"] in valid_labels
        assert data["color"] in valid_colors


# ── Inbound event schema (what .NET sends, what Python must parse) ─────────────
# C# publishes with Wolverine's default camelCase JSON.
# Pydantic must parse both field variations robustly.

def test_delivery_anomaly_event_parses_camelcase_json():
    """Simulates the JSON .NET sends via Wolverine → Python must parse it."""
    raw = json.dumps({
        "deliveryId": str(uuid4()),
        "districtId": "mezzeh",
        "anomalyType": "StalePosition",
        "reason": "No GPS for 20 min",
        "driverLat": 33.505,
        "driverLng": 36.243,
        "occurredAt": datetime.now(timezone.utc).isoformat(),
    })
    event = DeliveryAnomalyIntegrationEvent.model_validate_json(raw)
    assert event.districtId == "mezzeh"
    assert event.anomalyType == "StalePosition"


def test_driver_position_event_parses_camelcase_json():
    """Simulates the JSON .NET sends via Wolverine → Python must parse it."""
    raw = json.dumps({
        "driverId": str(uuid4()),
        "districtId": "babtouma",
        "lat": 33.522,
        "lng": 36.307,
        "deliveryStatus": "InTransit",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    event = DriverPositionIntegrationEvent.model_validate_json(raw)
    assert event.districtId == "babtouma"
    assert event.deliveryStatus == "InTransit"
