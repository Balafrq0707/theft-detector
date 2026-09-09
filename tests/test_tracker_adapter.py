from theft_detector.adapters.tracker_adapter import (
    TrackerAdapter,
)
from theft_detector.models.detection import Detection


def test_tracker_output_is_converted_to_detection():

    adapter = TrackerAdapter()

    tracker_output = {
        "track_id": 10,
        "label": "parcel",
        "bbox": (100, 200, 300, 400),
    }

    detection = adapter.to_detection(
        tracker_output
    )

    assert isinstance(
        detection,
        Detection,
    )

    assert detection.track_id == 10
    assert detection.label == "parcel"
    assert detection.bbox == (
        100,
        200,
        300,
        400,
    )

def test_multiple_tracker_outputs_are_converted():

    adapter = TrackerAdapter()

    tracker_outputs = [
        {
            "track_id": 1,
            "label": "person",
            "bbox": (10, 20, 100, 200),
        },
        {
            "track_id": 10,
            "label": "parcel",
            "bbox": (120, 220, 170, 280),
        },
    ]

    detections = adapter.to_detections(
        tracker_outputs
    )

    assert len(detections) == 2

    assert detections[0].track_id == 1
    assert detections[0].label == "person"

    assert detections[1].track_id == 10
    assert detections[1].label == "parcel"

def test_tracker_outputs_generator_is_converted():

    adapter = TrackerAdapter()

    tracker_outputs = (
        output
        for output in [
            {
                "track_id": 1,
                "label": "person",
                "bbox": (10, 20, 100, 200),
            },
            {
                "track_id": 10,
                "label": "parcel",
                "bbox": (120, 220, 170, 280),
            },
        ]
    )

    detections = adapter.to_detections(
        tracker_outputs
    )

    assert len(detections) == 2
    assert detections[0].track_id == 1
    assert detections[1].track_id == 10

def test_empty_tracker_outputs_returns_empty_list():

    adapter = TrackerAdapter()

    detections = adapter.to_detections([])

    assert detections == []