from theft_detector.config import TheftDetectorConfig
from theft_detector.models.event import (
    EventType,
    TheftEvent,
)
from theft_detector.models.state import ParcelState
from theft_detector.tracking.registry import ParcelRecord


class TheftReasoningEngine:

    def __init__(
        self,
        config: TheftDetectorConfig,
    ):

        self.config = config

    def update_missing_state(
        self,
        parcel: ParcelRecord,
        timestamp: float,
    ) -> None:

        # A reported parcel should keep its final state
        # until it is explicitly seen again.
        if (
            parcel.state
            == ParcelState.SUSPICIOUSLY_REMOVED
        ):
            return

        time_missing = (
            timestamp - parcel.last_seen
        )

        if (
            time_missing >=
            self.config.parcel_missing_timeout
        ):

            # Remember when the parcel entered
            # the missing state.
            if parcel.missing_since is None:
                parcel.missing_since = timestamp

            parcel.state = ParcelState.MISSING

    def calculate_risk(
        self,
        parcel: ParcelRecord,
        timestamp: float,
    ) -> float:

        # No interaction means there is currently
        # no evidence connecting the disappearance
        # to a person.
        if parcel.last_interaction_time is None:
            return 0.0

        # Risk is calculated only while the parcel
        # is actively considered missing.
        if (
            parcel.state
            != ParcelState.MISSING
        ):
            return 0.0

        # The parcel must have a known missing time.
        if parcel.missing_since is None:
            return 0.0

        # Wait and observe before generating
        # suspicion about the disappearance.
        observation_duration = (
            timestamp - parcel.missing_since
        )

        if (
            observation_duration
            < self.config.observation_timeout
        ):
            return 0.0

        missing_duration = (
            timestamp - parcel.last_seen
        )

        # Measure how recently the interaction
        # happened before the parcel became missing.
        interaction_recency = (
            parcel.missing_since
            - parcel.last_interaction_time
        )

        risk = 0.0

        # Parcel became missing shortly after
        # the last interaction.
        if interaction_recency <= 10.0:
            risk += 0.45

        # Parcel has remained missing long enough.
        if (
            missing_duration
            >= self.config.parcel_missing_timeout
        ):
            risk += 0.25

        # A completed interaction lasted long enough
        # to contribute to the risk score.
        if (
            parcel.last_interaction_duration
            is not None
            and parcel.last_interaction_duration
            >= self.config.minimum_interaction_duration
        ):
            risk += 0.20

        # A tracked person was associated with
        # the most recent completed interaction.
        if (
            parcel.last_interacting_person_id
            is not None
        ):
            risk += 0.10

        return min(risk, 1.0)

    def create_event(
        self,
        parcel: ParcelRecord,
        timestamp: float,
    ) -> TheftEvent | None:

        risk_score = self.calculate_risk(
            parcel,
            timestamp,
        )

        if (
            risk_score
            < self.config.suspicious_risk_threshold
        ):
            return None

        # Prevent duplicate events during
        # the configured cooldown period.
        if (
            parcel.event_reported_at
            is not None
        ):

            elapsed = (
                timestamp
                - parcel.event_reported_at
            )

            if (
                elapsed
                < self.config.event_cooldown
            ):
                return None

        if (
            risk_score
            >= self.config.theft_risk_threshold
        ):

            event_type = (
                EventType.POSSIBLE_PARCEL_THEFT
            )

        else:

            event_type = (
                EventType.SUSPICIOUS_PARCEL_REMOVAL
            )

        parcel.event_reported_at = timestamp

        parcel.state = (
            ParcelState.SUSPICIOUSLY_REMOVED
        )

        return TheftEvent(
            event_type=event_type,
            parcel_id=parcel.track_id,
            person_id=(
                parcel.last_interacting_person_id
            ),
            timestamp=timestamp,
            risk_score=risk_score,
            reason=(
                "Parcel disappeared after "
                "person interaction"
            ),
        )