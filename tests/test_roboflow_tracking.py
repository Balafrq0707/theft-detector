from theft_detector.adapters.roboflow_adapter import (
    RoboflowAdapter,
)
from theft_detector.tracking.simple_tracker import (
    SimpleTracker,
)


def test_roboflow_detections_are_tracked_across_frames():
    adapter = RoboflowAdapter()
    tracker = SimpleTracker()

    first_result = {
        "predictions": {
            "image": {
                "width": 720,
                "height": 1280,
            },
            "predictions": [
                {
                    "width": 200.0,
                    "height": 200.0,
                    "x": 300.0,
                    "y": 500.0,
                    "confidence": 0.90,
                    "class_id": 0,
                    "class": "parcel",
                }
            ],
        }
    }

    second_result = {
        "predictions": {
            "image": {
                "width": 720,
                "height": 1280,
            },
            "predictions": [
                {
                    "width": 200.0,
                    "height": 200.0,
                    "x": 310.0,
                    "y": 505.0,
                    "confidence": 0.88,
                    "class_id": 0,
                    "class": "parcel",
                }
            ],
        }
    }

    first_detections = adapter.to_detections(
        first_result
    )

    second_detections = adapter.to_detections(
        second_result
    )

    first_tracked = tracker.update(
        first_detections
    )

    second_tracked = tracker.update(
        second_detections
    )

    assert first_tracked[0]["track_id"] == 1
    assert second_tracked[0]["track_id"] == 1

    assert second_tracked[0]["label"] == "parcel"
    assert second_tracked[0]["confidence"] == 0.88

from theft_detector.adapters.tracker_adapter import (
    TrackerAdapter,
)


def test_roboflow_tracking_converts_to_detection_objects():
    roboflow_adapter = RoboflowAdapter()
    tracker = SimpleTracker()
    tracker_adapter = TrackerAdapter()

    result = {
        "predictions": {
            "image": {
                "width": 720,
                "height": 1280,
            },
            "predictions": [
                {
                    "width": 200.0,
                    "height": 200.0,
                    "x": 300.0,
                    "y": 500.0,
                    "confidence": 0.90,
                    "class_id": 0,
                    "class": "parcel",
                },
                {
                    "width": 100.0,
                    "height": 200.0,
                    "x": 500.0,
                    "y": 400.0,
                    "confidence": 0.85,
                    "class_id": 1,
                    "class": "person",
                },
            ],
        }
    }

    raw_detections = (
        roboflow_adapter.to_detections(result)
    )

    tracked_detections = tracker.update(
        raw_detections
    )

    detections = tracker_adapter.to_detections(
        tracked_detections
    )

    assert len(detections) == 2

    parcel = detections[0]
    person = detections[1]

    assert parcel.track_id == 1
    assert parcel.label == "parcel"
    assert parcel.bbox == (
        200.0,
        400.0,
        400.0,
        600.0,
    )

    assert person.track_id == 2
    assert person.label == "person"
    assert person.bbox == (
        450.0,
        300.0,
        550.0,
        500.0,
    )