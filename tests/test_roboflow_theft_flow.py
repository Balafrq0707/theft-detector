from theft_detector.adapters.roboflow_adapter import (
    RoboflowAdapter,
)
from theft_detector.adapters.tracker_adapter import (
    TrackerAdapter,
)
from theft_detector.config import TheftDetectorConfig
from theft_detector.detector import TheftDetector
from theft_detector.tracking.simple_tracker import (
    SimpleTracker,
)


def test_roboflow_pipeline_detects_parcel_disappearance():

    roboflow_adapter = RoboflowAdapter()
    tracker = SimpleTracker()
    tracker_adapter = TrackerAdapter()

    config = TheftDetectorConfig(
        parcel_missing_timeout=2.0,
        observation_timeout=1.0,
        minimum_interaction_duration=0.5,
    )

    detector = TheftDetector(config=config)

    # --------------------------------------------------
    # Frame 1: person and parcel are together.
    # --------------------------------------------------

    result_1 = {
        "predictions": {
            "image": {
                "width": 720,
                "height": 1280,
            },
            "predictions": [
                {
                    "width": 100.0,
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
                    "x": 300.0,
                    "y": 500.0,
                    "confidence": 0.90,
                    "class_id": 1,
                    "class": "person",
                },
            ],
        }
    }

    detections = roboflow_adapter.to_detections(
        result_1
    )

    tracked = tracker.update(detections)

    detections = tracker_adapter.to_detections(
        tracked
    )

    events = detector.update(
        detections=detections,
        timestamp=0.0,
    )

    assert events == []

    # --------------------------------------------------
    # Frame 2: interaction continues.
    # --------------------------------------------------

    result_2 = {
        "predictions": {
            "image": {
                "width": 720,
                "height": 1280,
            },
            "predictions": [
                {
                    "width": 100.0,
                    "height": 200.0,
                    "x": 305.0,
                    "y": 505.0,
                    "confidence": 0.90,
                    "class_id": 0,
                    "class": "parcel",
                },
                {
                    "width": 100.0,
                    "height": 200.0,
                    "x": 305.0,
                    "y": 505.0,
                    "confidence": 0.90,
                    "class_id": 1,
                    "class": "person",
                },
            ],
        }
    }

    detections = roboflow_adapter.to_detections(
        result_2
    )

    tracked = tracker.update(detections)

    detections = tracker_adapter.to_detections(
        tracked
    )

    events = detector.update(
        detections=detections,
        timestamp=1.0,
    )

    assert events == []

    # --------------------------------------------------
    # Frame 3: parcel disappears.
    # --------------------------------------------------

    result_3 = {
        "predictions": {
            "image": {
                "width": 720,
                "height": 1280,
            },
            "predictions": [
                {
                    "width": 100.0,
                    "height": 200.0,
                    "x": 305.0,
                    "y": 505.0,
                    "confidence": 0.90,
                    "class_id": 1,
                    "class": "person",
                },
            ],
        }
    }

    detections = roboflow_adapter.to_detections(
        result_3
    )

    tracked = tracker.update(detections)

    detections = tracker_adapter.to_detections(
        tracked
    )

    events = detector.update(
        detections=detections,
        timestamp=3.0,
    )

    assert events == []

    # --------------------------------------------------
    # Frame 4: parcel remains missing long enough.
    # --------------------------------------------------

    detections = []

    events = detector.update(
        detections=detections,
        timestamp=5.0,
    )

    assert len(events) == 1

    event = events[0]

    assert event.parcel_id == 1
    assert event.person_id == 2
    assert event.risk_score >= config.suspicious_risk_threshold