from dataclasses import dataclass, field
from typing import Optional

from theft_detector.models.detection import Detection
from theft_detector.models.state import ParcelState


@dataclass
class ParcelRecord:
    track_id: int

    state: ParcelState = ParcelState.UNKNOWN

    first_seen: float = 0.0
    last_seen: float = 0.0

    missing_since: Optional[float] = None

    last_interaction_time: Optional[float] = None
    last_interacting_person_id: Optional[int] = None
    interacting_person_id: Optional[int] = None
    interaction_started_at: Optional[float] = None

    event_reported_at: Optional[float] = None

    last_interaction_duration: Optional[float] = None

    # Historical interaction evidence
    last_interacting_person_id: Optional[int] = None

    last_interaction_time: Optional[float] = None

    last_interaction_duration: Optional[float] = None

@dataclass
class PersonRecord:
    track_id: int

    last_seen: float = 0.0


@dataclass
class ObjectRegistry:

    parcels: dict[int, ParcelRecord] = field(
        default_factory=dict
    )

    persons: dict[int, PersonRecord] = field(
        default_factory=dict
    )

    def update_person(
        self,
        detection: Detection,
        timestamp: float,
    ) -> None:

        person = self.persons.get(detection.track_id)

        if person is None:
            person = PersonRecord(
                track_id=detection.track_id,
                last_seen=timestamp,
            )

            self.persons[detection.track_id] = person

        else:
            person.last_seen = timestamp

    def update_parcel(
        self,
        detection: Detection,
        timestamp: float,
    ) -> ParcelRecord:

        parcel = self.parcels.get(detection.track_id)

        if parcel is None:

            parcel = ParcelRecord(
                track_id=detection.track_id,
                state=ParcelState.VISIBLE,
                first_seen=timestamp,
                last_seen=timestamp,
            )

            self.parcels[detection.track_id] = parcel

        else:

            parcel.last_seen = timestamp

            # The parcel is visible again, so it is no longer missing.
            parcel.missing_since = None

            # A new observation resets the current parcel state.
            parcel.state = ParcelState.VISIBLE