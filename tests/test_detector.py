from theft_detector import (
    Detection,
    TheftDetector,
)
from theft_detector.models.state import ParcelState
import pytest


def test_suspicious_parcel_removal():

    detector = TheftDetector()

    # Parcel and person are visible
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    # Continued interaction
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    # Parcel disappears
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            )
        ],
        timestamp=8.0,
    )

    # We are still observing, so no event yet
    assert len(events) == 0

    # Parcel remains missing after the observation period
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            )
        ],
        timestamp=18.0,
    )

    # Now an event should be generated
    assert len(events) == 1
    assert events[0].parcel_id == 10

def test_no_event_when_parcel_remains_visible():

    detector = TheftDetector()

    # Person and parcel are visible
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    assert len(events) == 0

    # They remain visible later
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=10.0,
    )

    assert len(events) == 0

def test_no_event_when_parcel_disappears_without_interaction():

    detector = TheftDetector()

    # Parcel is visible without any person nearby
    detector.update(
        detections=[
            Detection(
                track_id=10,
                label="parcel",
                bbox=(500, 500, 550, 550),
            ),
        ],
        timestamp=0.0,
    )

    # Parcel is no longer visible after the missing timeout
    events = detector.update(
        detections=[],
        timestamp=10.0,
    )

    assert len(events) == 0

def test_no_event_when_parcel_reappears():

    detector = TheftDetector()

    # Person and parcel interact
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    # Continued interaction
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    # Parcel temporarily disappears
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
        ],
        timestamp=8.0,
    )

    # Don't report yet — continue observing
    assert len(events) == 0

    # Parcel reappears
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=10.0,
    )

    assert len(events) == 0


def test_reported_parcel_state_is_preserved():

    detector = TheftDetector()

    # Person and parcel interact
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    # Continued interaction
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    # Parcel disappears and becomes missing
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            )
        ],
        timestamp=8.0,
    )

    # Parcel remains missing long enough for an event
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            )
        ],
        timestamp=18.0,
    )

    assert len(events) == 1

    # Another update arrives while the parcel is still missing
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            )
        ],
        timestamp=19.0,
    )

    # No duplicate event should be generated
    assert len(events) == 0

    # The parcel must remain in its reported state
    parcel = detector.registry.parcels[10]

    assert (
        parcel.state
        == ParcelState.SUSPICIOUSLY_REMOVED
    )

def test_reported_parcel_can_reappear():

    detector = TheftDetector()

    # Person and parcel interact
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    # Continued interaction
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    # Parcel disappears
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
        ],
        timestamp=8.0,
    )

    # Parcel remains missing long enough to generate an event
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
        ],
        timestamp=18.0,
    )

    assert len(events) == 1

    parcel = detector.registry.parcels[10]

    assert (
        parcel.state
        == ParcelState.SUSPICIOUSLY_REMOVED
    )

    # The parcel appears again
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=20.0,
    )

    # Reappearance itself should not generate another event
    assert len(events) == 0

    parcel = detector.registry.parcels[10]

    assert parcel.state == ParcelState.VISIBLE

    assert parcel.interacting_person_id == 1

def test_visible_parcel_can_have_active_interaction():

    detector = TheftDetector()

    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    parcel = detector.registry.parcels[10]

    # The parcel is physically visible
    assert parcel.state == ParcelState.VISIBLE

    # But it is also being interacted with
    assert parcel.interacting_person_id == 1

    assert parcel.last_interaction_time == 0.0

def test_interaction_ends_when_person_moves_away():

    detector = TheftDetector()

    # Person is interacting with the parcel
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    parcel = detector.registry.parcels[10]

    assert parcel.interacting_person_id == 1
    assert parcel.last_interaction_time == 0.0

    # Person moves far away from the parcel
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    parcel = detector.registry.parcels[10]

    # The parcel is still visible
    assert parcel.state == ParcelState.VISIBLE

    # But there is no active interaction anymore
    assert parcel.interacting_person_id is None

    # Historical interaction evidence is preserved
    assert parcel.last_interaction_time == 0.0

def test_new_interaction_gets_new_start_time():

    detector = TheftDetector()

    # First interaction starts
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    parcel = detector.registry.parcels[10]

    assert parcel.interaction_started_at == 0.0

    # Person moves away
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    parcel = detector.registry.parcels[10]

    # Active interaction has ended
    assert parcel.interacting_person_id is None

    # A future interaction must get a fresh start time
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=10.0,
    )

    parcel = detector.registry.parcels[10]

    assert parcel.interacting_person_id == 1
    assert parcel.interaction_started_at == 10.0

def test_last_interacting_person_is_preserved():

    detector = TheftDetector()

    # Person interacts with the parcel
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    parcel = detector.registry.parcels[10]

    assert parcel.interacting_person_id == 1

    # Person moves away
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    parcel = detector.registry.parcels[10]

    # No active interaction anymore
    assert parcel.interacting_person_id is None

    # But historical evidence must remain
    assert parcel.last_interacting_person_id == 1
    assert parcel.last_interaction_time == 0.0

def test_theft_event_uses_last_interacting_person():

    detector = TheftDetector()

    # Person interacts with the parcel
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    # Continued interaction
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    # Person moves away while parcel remains visible
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=3.0,
    )

    parcel = detector.registry.parcels[10]

    assert parcel.interacting_person_id is None
    assert parcel.last_interacting_person_id == 1

    # Parcel disappears
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
        ],
        timestamp=8.0,
    )

    # Theft event is generated
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
        ],
        timestamp=18.0,
    )

    assert len(events) == 1

    assert events[0].person_id == 1

def test_completed_interaction_duration_is_preserved():

    detector = TheftDetector()

    # Interaction begins
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    # Interaction continues
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=4.0,
    )

    # Person moves away
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=5.0,
    )

    parcel = detector.registry.parcels[10]

    assert parcel.interacting_person_id is None

    # Completed interaction lasted from 0s to 4s
    assert parcel.last_interaction_duration == 4.0

def test_completed_interaction_duration_contributes_to_risk():

    detector = TheftDetector()

    # Interaction begins
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    # Interaction continues long enough
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    # Interaction ends
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=3.0,
    )

    parcel = detector.registry.parcels[10]

    assert parcel.last_interaction_duration == 2.0

    # Parcel disappears and enters the missing state.
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
        ],
        timestamp=8.0,
    )

    # Continue observing the parcel while it remains missing.
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
        ],
        timestamp=18.0,
    )

    assert len(events) == 1

    # Expected risk:
    # 0.45 -> parcel became missing shortly after interaction
    # 0.25 -> parcel remained missing
    # 0.20 -> completed interaction duration
    # 0.10 -> known last interacting person
    assert events[0].risk_score == pytest.approx(1.0)

def test_short_completed_interaction_does_not_contribute_to_risk():

    detector = TheftDetector()

    # Interaction begins
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    # Interaction ends before the minimum duration
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.5,
    )

    parcel = detector.registry.parcels[10]

    assert parcel.last_interaction_duration == 0.5

    # Parcel disappears
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
        ],
        timestamp=6.0,
    )

    # Continue observing the missing parcel
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
        ],
        timestamp=16.0,
    )

    assert len(events) == 1

    # Expected risk:
    # 0.45 -> parcel became missing shortly after interaction
    # 0.25 -> parcel remained missing
    # 0.00 -> interaction was shorter than minimum duration
    # 0.10 -> known last interacting person
    assert events[0].risk_score == pytest.approx(0.80)

    def test_new_person_interaction_replaces_last_interacting_person():

        detector = TheftDetector()

        # Person 1 interacts with the parcel
        detector.update(
            detections=[
                Detection(
                    track_id=1,
                    label="person",
                    bbox=(100, 100, 200, 300),
                ),
                Detection(
                    track_id=10,
                    label="parcel",
                    bbox=(120, 220, 170, 280),
                ),
            ],
            timestamp=0.0,
        )

        # Person 1 moves away
        detector.update(
            detections=[
                Detection(
                    track_id=1,
                    label="person",
                    bbox=(1000, 1000, 1100, 1200),
                ),
                Detection(
                    track_id=10,
                    label="parcel",
                    bbox=(120, 220, 170, 280),
                ),
            ],
            timestamp=2.0,
        )

        parcel = detector.registry.parcels[10]

        assert parcel.interacting_person_id is None
        assert parcel.last_interacting_person_id == 1

        # Person 2 now interacts with the same parcel
        detector.update(
            detections=[
                Detection(
                    track_id=2,
                    label="person",
                    bbox=(100, 100, 200, 300),
                ),
                Detection(
                    track_id=10,
                    label="parcel",
                    bbox=(120, 220, 170, 280),
                ),
            ],
            timestamp=3.0,
        )

        parcel = detector.registry.parcels[10]

        # Person 2 is the active interacting person
        assert parcel.interacting_person_id == 2

        # Historical evidence should now point to Person 2
        assert parcel.last_interacting_person_id == 2

def test_new_person_replaces_last_interacting_person():

    detector = TheftDetector()

    # Person 1 interacts with the parcel
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    # Person 1 moves away
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=3.0,
    )

    # Person 2 starts interacting with the same parcel
    detector.update(
        detections=[
            Detection(
                track_id=2,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=4.0,
    )

    parcel = detector.registry.parcels[10]

    # Person 2 is currently interacting
    assert parcel.interacting_person_id == 2

    # Person 2 should now be the latest interacting person
    assert parcel.last_interacting_person_id == 2

    # A new interaction must have its own start time
    assert parcel.interaction_started_at == 4.0

def test_new_interaction_resets_previous_duration():

    detector = TheftDetector()

    # First interaction begins
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    # First interaction ends
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    parcel = detector.registry.parcels[10]

    # First interaction duration
    assert parcel.last_interaction_duration == 2.0
    assert parcel.interacting_person_id is None

    # A new interaction begins later
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=10.0,
    )

    parcel = detector.registry.parcels[10]

    # New interaction must have a new start time
    assert parcel.interaction_started_at == 10.0

    # Active interaction should be associated with the person
    assert parcel.interacting_person_id == 1

    # The previous completed duration is still historical data
    assert parcel.last_interaction_duration == 2.0

def test_old_interaction_does_not_create_high_risk():

    detector = TheftDetector()

    # Person interacts with parcel
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    # Interaction continues
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    # Person moves away
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=3.0,
    )

    # Much later, parcel disappears
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
        ],
        timestamp=20.0,
    )

    # Old interaction should not immediately create
    # a theft event.
    assert len(events) == 0 

def test_new_interaction_resets_previous_completed_duration():

    detector = TheftDetector()

    # First interaction begins
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    # First interaction continues
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=4.0,
    )

    # First interaction ends
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=5.0,
    )

    parcel = detector.registry.parcels[10]

    assert parcel.last_interaction_duration == 4.0

    # A new person starts interacting
    detector.update(
        detections=[
            Detection(
                track_id=2,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=10.0,
    )

    parcel = detector.registry.parcels[10]

    # New interaction should become active
    assert parcel.interacting_person_id == 2

    # New interaction starts fresh
    assert parcel.interaction_started_at == 10.0

def test_new_interaction_replaces_last_interacting_person():

    detector = TheftDetector()

    # Person 1 interacts with the parcel
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    # Person 1 moves away
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    # Person 2 interacts with the same parcel
    detector.update(
        detections=[
            Detection(
                track_id=2,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=4.0,
    )

    parcel = detector.registry.parcels[10]

    # Person 2 should now be the latest interacting person
    assert parcel.last_interacting_person_id == 2

    # Person 2 moves away and parcel disappears
    detector.update(
        detections=[
            Detection(
                track_id=2,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
        ],
        timestamp=10.0,
    )

    # Continue observing while the parcel remains missing
    events = detector.update(
        detections=[
            Detection(
                track_id=2,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
        ],
        timestamp=20.0,
    )

    assert len(events) == 1

    # Event must point to Person 2
    assert events[0].person_id == 2

def test_parcel_reappearing_resets_missing_evidence():

    detector = TheftDetector()

    # Person interacts with parcel
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    # Interaction continues
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    # Parcel disappears
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
        ],
        timestamp=8.0,
    )

    parcel = detector.registry.parcels[10]

    assert parcel.state == ParcelState.MISSING

    # Parcel reappears
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=9.0,
    )

    assert len(events) == 0

    parcel = detector.registry.parcels[10]

    # Parcel should no longer be missing
    assert parcel.state == ParcelState.VISIBLE

    # Missing evidence should be cleared
    assert parcel.missing_since is None

def test_old_interaction_does_not_contribute_to_risk():

    detector = TheftDetector()

    # Person interacts with the parcel
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    # Interaction continues
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    # Person moves away, but parcel remains visible
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=3.0,
    )

    # Much later, the parcel disappears
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
        ],
        timestamp=20.0,
    )

    # Continue observing while missing
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
        ],
        timestamp=30.0,
    )

    # The interaction was too old to strongly connect
    # it to the disappearance.
    assert len(events) == 0

def test_most_recent_interacting_person_is_used_for_event():

    detector = TheftDetector()

    # Person 1 interacts with the parcel
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    # Person 1 moves away
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    # Person 2 interacts with the same parcel
    detector.update(
        detections=[
            Detection(
                track_id=2,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=3.0,
    )

    # Person 2 continues interaction
    detector.update(
        detections=[
            Detection(
                track_id=2,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=5.0,
    )

    # Parcel disappears
    detector.update(
        detections=[
            Detection(
                track_id=2,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
        ],
        timestamp=10.0,
    )

    # Continue observing the missing parcel
    events = detector.update(
        detections=[
            Detection(
                track_id=2,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
        ],
        timestamp=20.0,
    )

    assert len(events) == 1

    # The event should point to Person 2,
    # the most recent person who interacted.
    assert events[0].person_id == 2

def test_old_interaction_does_not_contribute_to_theft_risk():

    detector = TheftDetector()

    # Person interacts with the parcel
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    # Interaction continues
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    # Person moves away, parcel remains visible
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=3.0,
    )

    # Parcel remains visible for a long time
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=20.0,
    )

    # Parcel finally disappears much later
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
        ],
        timestamp=30.0,
    )

    # The earlier interaction is too old to be
    # considered evidence for theft.
    assert len(events) == 0

def test_reappearing_parcel_resets_missing_history():

    detector = TheftDetector()

    # Person interacts with parcel
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    # Interaction continues
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    # Parcel disappears
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
        ],
        timestamp=8.0,
    )

    # Parcel reappears
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=10.0,
    )

    parcel = detector.registry.parcels[10]

    # Reappearance must clear the old missing history
    assert parcel.missing_since is None

def test_only_interacted_parcel_generates_theft_event():

    detector = TheftDetector()

    # Person interacts only with parcel 10.
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
            Detection(
                track_id=20,
                label="parcel",
                bbox=(800, 800, 850, 860),
            ),
        ],
        timestamp=0.0,
    )

    # Interaction continues.
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
            Detection(
                track_id=20,
                label="parcel",
                bbox=(800, 800, 850, 860),
            ),
        ],
        timestamp=2.0,
    )

    # Both parcels disappear.
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
        ],
        timestamp=8.0,
    )

    # Keep observing while both remain missing.
    events = detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
        ],
        timestamp=18.0,
    )

    # Only parcel 10 had person interaction history.
    assert len(events) == 1

    assert events[0].parcel_id == 10

    parcel_10 = detector.registry.parcels[10]
    parcel_20 = detector.registry.parcels[20]

    assert (
        parcel_10.state
        == ParcelState.SUSPICIOUSLY_REMOVED
    )

    assert (
        parcel_20.state
        == ParcelState.MISSING
    )

def test_one_person_can_interact_with_multiple_parcels():

    detector = TheftDetector()

    # One person is close enough to both parcels.
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 300, 400),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
            Detection(
                track_id=20,
                label="parcel",
                bbox=(200, 220, 250, 280),
            ),
        ],
        timestamp=0.0,
    )

    parcel_10 = detector.registry.parcels[10]
    parcel_20 = detector.registry.parcels[20]

    # Both parcels should have their own interaction
    # with the same tracked person.
    assert parcel_10.interacting_person_id == 1
    assert parcel_20.interacting_person_id == 1

    assert parcel_10.last_interacting_person_id == 1
    assert parcel_20.last_interacting_person_id == 1

    assert parcel_10.interaction_started_at == 0.0
    assert parcel_20.interaction_started_at == 0.0

    # Both parcels remain physically visible.
    assert parcel_10.state == ParcelState.VISIBLE
    assert parcel_20.state == ParcelState.VISIBLE

def test_multiple_people_interacting_with_same_parcel():

    detector = TheftDetector()

    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=2,
                label="person",
                bbox=(150, 100, 250, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(140, 220, 190, 280),
            ),
        ],
        timestamp=0.0,
    )

    parcel = detector.registry.parcels[10]

    # The parcel should have an active interaction
    # with one of the nearby tracked people.
    assert parcel.interacting_person_id in {1, 2}

    # The historical person should match the active one.
    assert (
        parcel.last_interacting_person_id
        == parcel.interacting_person_id
    )

    assert parcel.last_interaction_time == 0.0

    # Parcel is still physically visible.
    assert parcel.state == ParcelState.VISIBLE

def test_closest_person_is_selected_for_parcel_interaction():

    detector = TheftDetector()

    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(0, 0, 100, 200),
            ),
            Detection(
                track_id=2,
                label="person",
                bbox=(200, 0, 300, 200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(190, 150, 240, 190),
            ),
        ],
        timestamp=0.0,
    )

    parcel = detector.registry.parcels[10]

    # Person 2 is closer to the parcel.
    assert parcel.interacting_person_id == 2

    assert parcel.last_interacting_person_id == 2

    assert parcel.last_interaction_time == 0.0

def test_parcel_interaction_switches_to_new_closest_person():

    detector = TheftDetector()

    # Person 1 starts closest to the parcel.
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=2,
                label="person",
                bbox=(500, 100, 600, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    parcel = detector.registry.parcels[10]

    assert parcel.interacting_person_id == 1

    # Person 1 moves away.
    # Person 2 becomes the closest interacting person.
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=2,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=2.0,
    )

    parcel = detector.registry.parcels[10]

    # Active interaction should now belong to person 2.
    assert parcel.interacting_person_id == 2

    # Historical interaction should also update.
    assert parcel.last_interacting_person_id == 2

    assert parcel.last_interaction_time == 2.0

def test_switching_person_starts_new_interaction():

    detector = TheftDetector()

    # Person 1 interacts with the parcel.
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=2,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=0.0,
    )

    parcel = detector.registry.parcels[10]

    assert parcel.interacting_person_id == 1
    assert parcel.interaction_started_at == 0.0

    # Person 1 moves away.
    # Person 2 starts interacting.
    detector.update(
        detections=[
            Detection(
                track_id=1,
                label="person",
                bbox=(1000, 1000, 1100, 1200),
            ),
            Detection(
                track_id=2,
                label="person",
                bbox=(100, 100, 200, 300),
            ),
            Detection(
                track_id=10,
                label="parcel",
                bbox=(120, 220, 170, 280),
            ),
        ],
        timestamp=5.0,
    )

    parcel = detector.registry.parcels[10]

    # Person 2 is now actively interacting.
    assert parcel.interacting_person_id == 2

    # This must be a new interaction.
    assert parcel.interaction_started_at == 5.0

    # Historical attribution should now reflect Person 2.
    assert parcel.last_interacting_person_id == 2

    assert parcel.last_interaction_time == 5.0