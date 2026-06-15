from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DeliveryAnomalyIntegrationEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    deliveryId: UUID
    districtId: str
    anomalyType: str   # "EtaExceeded" | "RouteDeviation" | "StalePosition" | "UnexpectedStop"
    reason: str
    driverLat: float
    driverLng: float
    occurredAt: datetime


class DriverPositionIntegrationEvent(BaseModel):
    driverId: UUID
    districtId: str
    lat: float
    lng: float
    deliveryStatus: str
    timestamp: datetime


class DeliveryCompletedIntegrationEvent(BaseModel):
    deliveryId: UUID
    driverId: UUID
    districtId: str
    pickedUpAt: datetime
    deliveredAt: datetime
    actualDurationSeconds: float
    expectedDurationSeconds: float


class UrgencyResultMessage(BaseModel):
    deliveryId: str
    urgencyScore: int
    aiNote: str


class ForecastResultMessage(BaseModel):
    districtId: str
    expectedDeliveries: int
    staffingRatio: float
    label: str
    color: str
    generatedAt: str


class ChatBody(BaseModel):
    question: str
    context: dict


class CandidateContext(BaseModel):
    rank: int
    name: str
    distance_m: float
    on_time_rate_pct: float | None = None
    score: float


class RecommendationRequest(BaseModel):
    delivery_id: str
    district_id: str
    anomaly_type: str | None = None
    anomaly_reason: str | None = None
    candidates: list[CandidateContext]


class RecommendationResponse(BaseModel):
    recommended_action: str   # Reassign | Contact | Cancel | Monitor
    candidate_rank: int | None = None   # 1–3, None when action doesn't target a driver
    reason: str
    urgency_score: int        # 1–10


class DemandSurgeMessage(BaseModel):
    districtId: str
    currentCount: int
    historicalMean: float
    deviations: float          # z-score above baseline
    detectedAt: str


class AnomalyIncidentMessage(BaseModel):
    districtId: str
    anomalyCount: int
    windowMinutes: int
    summary: str               # Groq-generated one-liner
    detectedAt: str


class StaffingRequest(BaseModel):
    district: str
    target_datetime: str       # ISO string
    day_of_week: int           # 0=Mon … 6=Sun
    target_hour: int           # 0–23
    historical_avg_deliveries: float
    recent_surge_detected: bool


class StaffingResponse(BaseModel):
    recommended_drivers: int
    confidence: str            # "high" | "medium" | "low"
    reasoning: str
