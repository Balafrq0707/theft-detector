from typing import Iterable

from theft_detector.config import TheftDetectorConfig
from theft_detector.interaction import InteractionAnalyzer
from theft_detector.models.detection import Detection
from theft_detector.models.event import TheftEvent
from theft_detector.reasoning import TheftReasoningEngine
from theft_detector.tracking import ObjectRegistry


class TheftDetector:

    def __init__(
        self,
        config: TheftDetectorConfig | None = None,
    ):

        self.config = (
            config or TheftDetectorConfig()
        )

        self.registry = ObjectRegistry()

        self.interaction_analyzer = (
            InteractionAnalyzer(
                interaction_distance=
                self.config.interaction_distance
            )
        )

        self.reasoning_engine = (
            TheftReasoningEngine(
                self.config
            )
        )

    def update(
        self,
        detections: Iterable[Detection],
        timestamp: float,
    ) -> list[TheftEvent]:

        detections = list(detections)

        persons = [
            detection
            for detection in detections
            if detection.label == "person"
        ]

        parcels = [
            detection
            for detection in detections
            if detection.label == "parcel"
        ]

        # Update person history
        for person in persons:

            self.registry.update_person(
                detection=person,
                timestamp=timestamp,
            )

        # Update parcel visibility/history
        for parcel_detection in parcels:

            self.registry.update_parcel(
                detection=parcel_detection,
                timestamp=timestamp,
            )

        # Detect current person-parcel interactions
        interactions = (
            self.interaction_analyzer
            .find_interactions(
                persons=persons,
                parcels=parcels,
            )
        )

        active_interaction_parcel_ids = {
            interaction.parcel_id
            for interaction in interactions
        }

        # End interactions for visible parcels that
        # are no longer interacting with a person.
        for parcel_detection in parcels:

            parcel = self.registry.parcels.get(
                parcel_detection.track_id
            )

            if parcel is None:
                continue

            if (
                parcel.track_id
                in active_interaction_parcel_ids
            ):
                continue

            if (
                parcel.interacting_person_id
                is not None
                and parcel.interaction_started_at
                is not None
            ):

                # Normally use the last timestamp where
                # interaction was actually observed.
                end_time = (
                    parcel.last_interaction_time
                )

                # If the interaction began on the previous
                # update and immediately ends now, use the
                # current timestamp as the end.
                if (
                    end_time
                    == parcel.interaction_started_at
                ):
                    end_time = timestamp

                parcel.last_interaction_duration = (
                    end_time
                    - parcel.interaction_started_at
                )

                parcel.interacting_person_id = None
                parcel.interaction_started_at = None

        # Record current interactions
        for interaction in interactions:

            parcel = (
                self.registry.parcels.get(
                    interaction.parcel_id
                )
            )

            if parcel is None:
                continue

            previous_person_id = (
                parcel.interacting_person_id
            )

            # A different person means the previous
            # interaction has ended and a new interaction
            # begins.
            if (
                previous_person_id is not None
                and previous_person_id
                != interaction.person_id
            ):

                # Preserve the completed duration of the
                # previous person's interaction.
                if (
                    parcel.interaction_started_at
                    is not None
                    and parcel.last_interaction_time
                    is not None
                ):

                    parcel.last_interaction_duration = (
                        parcel.last_interaction_time
                        - parcel.interaction_started_at
                    )

                # Start the new person's interaction.
                parcel.interaction_started_at = (
                    timestamp
                )

            # No active interaction means this is a
            # completely new interaction.
            elif (
                previous_person_id is None
            ):

                parcel.interaction_started_at = (
                    timestamp
                )

            # Active interaction
            parcel.interacting_person_id = (
                interaction.person_id
            )

            # Historical interaction evidence
            parcel.last_interacting_person_id = (
                interaction.person_id
            )

            parcel.last_interaction_time = (
                timestamp
            )

        # Update missing state for parcels that are
        # no longer visible.
        for parcel in (
            self.registry.parcels.values()
        ):

            self.reasoning_engine.update_missing_state(
                parcel=parcel,
                timestamp=timestamp,
            )

        # Generate theft-related events
        events: list[TheftEvent] = []

        for parcel in (
            self.registry.parcels.values()
        ):

            event = (
                self.reasoning_engine.create_event(
                    parcel=parcel,
                    timestamp=timestamp,
                )
            )

            if event is not None:

                events.append(event)

        return events