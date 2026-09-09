from theft_detector.adapters.roboflow_adapter import (
    RoboflowAdapter,
)
from theft_detector.adapters.tracker_adapter import (
    TrackerAdapter,
)
from theft_detector.detector import TheftDetector
from theft_detector.tracking.simple_tracker import (
    SimpleTracker,
)


def test_roboflow_detections_reach_theft_detector():
    roboflow_adapter = RoboflowAdapter()
    tracker = SimpleTracker()
    tracker_adapter = TrackerAdapter()
    detector = TheftDetector()

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

    events = detector.update(
        detections=detections,
        timestamp=0.0,
    )

    assert events == []

    assert len(detector.registry.parcels) == 1
    assert len(detector.registry.persons) == 1