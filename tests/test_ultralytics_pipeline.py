from theft_detector.adapters.ultralytics_adapter import (
    UltralyticsAdapter,
)
from theft_detector.detector import TheftDetector
from theft_detector.pipeline import TheftDetectionPipeline


def test_ultralytics_output_flows_into_pipeline():

    class MockBoxes:

        def __init__(self):
            self.id = [1, 10]
            self.cls = [0, 1]
            self.xyxy = [
                [100, 100, 200, 300],
                [120, 220, 170, 280],
            ]

    class MockResult:

        def __init__(self):
            self.boxes = MockBoxes()

    detector = TheftDetector()

    pipeline = TheftDetectionPipeline(
        detector=detector
    )

    ultralytics_adapter = UltralyticsAdapter()

    result = MockResult()

    # Convert Ultralytics result
    # into tracker-compatible output.
    tracker_outputs = (
        ultralytics_adapter
        .result_to_tracker_outputs(result)
    )

    # Send converted output into the pipeline.
    events = pipeline.process(
        tracker_outputs=tracker_outputs,
        timestamp=0.0,
    )

    # An interaction exists, but nothing suspicious
    # has happened yet.
    assert events == []

    # Verify the objects successfully entered
    # the detector through the full pipeline.
    assert 1 in detector.registry.persons
    assert 10 in detector.registry.parcels

import pytest

from theft_detector.adapters.ultralytics_adapter import (
    UltralyticsAdapter,
)
from theft_detector.detector import TheftDetector
from theft_detector.models.event import EventType
from theft_detector.pipeline import TheftDetectionPipeline


def test_ultralytics_results_produce_theft_event():

    class MockBoxes:

        def __init__(
            self,
            track_ids,
            class_ids,
            boxes,
        ):
            self.id = track_ids
            self.cls = class_ids
            self.xyxy = boxes

    class MockResult:

        def __init__(
            self,
            track_ids,
            class_ids,
            boxes,
        ):
            self.boxes = MockBoxes(
                track_ids,
                class_ids,
                boxes,
            )

    detector = TheftDetector()

    pipeline = TheftDetectionPipeline(
        detector=detector
    )

    ultralytics_adapter = UltralyticsAdapter()

    def process_result(
        result,
        timestamp,
    ):
        tracker_outputs = (
            ultralytics_adapter
            .result_to_tracker_outputs(result)
        )

        return pipeline.process(
            tracker_outputs=tracker_outputs,
            timestamp=timestamp,
        )

    # Frame 1: Person begins interacting.
    events = process_result(
        MockResult(
            track_ids=[1, 10],
            class_ids=[0, 1],
            boxes=[
                [100, 100, 200, 300],
                [120, 220, 170, 280],
            ],
        ),
        timestamp=0.0,
    )

    assert events == []

    # Frame 2: Interaction continues.
    events = process_result(
        MockResult(
            track_ids=[1, 10],
            class_ids=[0, 1],
            boxes=[
                [100, 100, 200, 300],
                [120, 220, 170, 280],
            ],
        ),
        timestamp=2.0,
    )

    assert events == []

    # Frame 3: Person moves away.
    events = process_result(
        MockResult(
            track_ids=[1, 10],
            class_ids=[0, 1],
            boxes=[
                [1000, 1000, 1100, 1200],
                [120, 220, 170, 280],
            ],
        ),
        timestamp=3.0,
    )

    assert events == []

    # Frame 4: Parcel disappears.
    events = process_result(
        MockResult(
            track_ids=[1],
            class_ids=[0],
            boxes=[
                [1000, 1000, 1100, 1200],
            ],
        ),
        timestamp=8.0,
    )

    assert events == []

    # Frame 5: Parcel remains missing long enough.
    events = process_result(
        MockResult(
            track_ids=[1],
            class_ids=[0],
            boxes=[
                [1000, 1000, 1100, 1200],
            ],
        ),
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

    assert event.risk_score == pytest.approx(
        1.0
    )