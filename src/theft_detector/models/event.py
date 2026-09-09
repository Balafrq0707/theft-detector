from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    SUSPICIOUS_PARCEL_REMOVAL = "suspicious_parcel_removal"

    POSSIBLE_PARCEL_THEFT = "possible_parcel_theft"


@dataclass(frozen=True)
class TheftEvent:
    event_type: EventType

    parcel_id: int

    person_id: Optional[int]

    timestamp: float

    risk_score: float

    reason: str