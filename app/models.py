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
