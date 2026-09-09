from theft_detector.detector import TheftDetector
from theft_detector.models.event import EventType
from theft_detector.pipeline import TheftDetectionPipeline
import pytest

def test_pipeline_detects_possible_parcel_theft():

    detector = TheftDetector()

    pipeline = TheftDetectionPipeline(
        detector=detector
    )

    # Person starts interacting with the parcel.
    events = pipeline.process(
        tracker_outputs=[
            {
                "track_id": 1,
                "label": "person",
                "bbox": (100, 100, 200, 300),
            },
            {
                "track_id": 10,
                "label": "parcel",
                "bbox": (120, 220, 170, 280),
            },
        ],
        timestamp=0.0,
    )

    assert events == []

    # Interaction continues.
    events = pipeline.process(
        tracker_outputs=[
            {
                "track_id": 1,
                "label": "person",
                "bbox": (100, 100, 200, 300),
            },
            {
                "track_id": 10,
                "label": "parcel",
                "bbox": (120, 220, 170, 280),
            },
        ],
        timestamp=2.0,
    )

    assert events == []

    # Person moves away while parcel remains visible.
    # This completes the interaction.
    events = pipeline.process(
        tracker_outputs=[
            {
                "track_id": 1,
                "label": "person",
                "bbox": (1000, 1000, 1100, 1200),
            },
            {
                "track_id": 10,
                "label": "parcel",
                "bbox": (120, 220, 170, 280),
            },
        ],
        timestamp=3.0,
    )

    assert events == []

    # Parcel disappears.
    events = pipeline.process(
        tracker_outputs=[
            {
                "track_id": 1,
                "label": "person",
                "bbox": (1000, 1000, 1100, 1200),
            },
        ],
        timestamp=8.0,
    )

    assert events == []

    # Parcel remains missing long enough
    # for the observation period.
    events = pipeline.process(
        tracker_outputs=[
            {
                "track_id": 1,
                "label": "person",
                "bbox": (1000, 1000, 1100, 1200),
            },
        ],
        timestamp=18.0,
    )

    assert len(events) == 1

    event = events[0]

    assert (
        event.event_type
        == EventType.POSSIBLE_PARCEL_THEFT
    )

    assert event.parcel_id == 10
    assert event.person_id == 1
    assert event.risk_score == pytest.approx(1.0)