from theft_detector.tracking.simple_tracker import SimpleTracker


def test_tracker_assigns_track_id_to_new_detection():
    tracker = SimpleTracker()

    detections = [
        {
            "label": "parcel",
            "bbox": (100, 100, 200, 200),
            "confidence": 0.9,
        }
    ]

    tracked = tracker.update(detections)

    assert len(tracked) == 1

    assert tracked[0]["track_id"] == 1
    assert tracked[0]["label"] == "parcel"
    assert tracked[0]["bbox"] == (100, 100, 200, 200)
    assert tracked[0]["confidence"] == 0.9

def test_tracker_preserves_track_id_for_same_detection():
    tracker = SimpleTracker()

    first_frame = [
        {
            "label": "parcel",
            "bbox": (100, 100, 200, 200),
            "confidence": 0.9,
        }
    ]

    second_frame = [
        {
            "label": "parcel",
            "bbox": (105, 105, 205, 205),
            "confidence": 0.88,
        }
    ]

    first_tracked = tracker.update(first_frame)
    second_tracked = tracker.update(second_frame)

    assert first_tracked[0]["track_id"] == 1
    assert second_tracked[0]["track_id"] == 1


def test_tracker_assigns_different_ids_to_multiple_detections():
    tracker = SimpleTracker()

    detections = [
        {
            "label": "parcel",
            "bbox": (100, 100, 200, 200),
            "confidence": 0.9,
        },
        {
            "label": "parcel",
            "bbox": (400, 400, 500, 500),
            "confidence": 0.85,
        },
    ]

    tracked = tracker.update(detections)

    assert len(tracked) == 2

    assert tracked[0]["track_id"] == 1
    assert tracked[1]["track_id"] == 2

    assert (
        tracked[0]["track_id"]
        != tracked[1]["track_id"]
    )

def test_tracker_preserves_ids_for_multiple_moving_detections():
    tracker = SimpleTracker()

    first_frame = [
        {
            "label": "parcel",
            "bbox": (100, 100, 200, 200),
            "confidence": 0.9,
        },
        {
            "label": "parcel",
            "bbox": (400, 400, 500, 500),
            "confidence": 0.85,
        },
    ]

    second_frame = [
        {
            "label": "parcel",
            "bbox": (110, 105, 210, 205),
            "confidence": 0.88,
        },
        {
            "label": "parcel",
            "bbox": (410, 405, 510, 505),
            "confidence": 0.83,
        },
    ]

    first_tracked = tracker.update(first_frame)
    second_tracked = tracker.update(second_frame)

    assert first_tracked[0]["track_id"] == 1
    assert first_tracked[1]["track_id"] == 2

    assert second_tracked[0]["track_id"] == 1
    assert second_tracked[1]["track_id"] == 2

def test_tracker_preserves_ids_when_detection_order_changes():
    tracker = SimpleTracker()

    first_frame = [
        {
            "label": "parcel",
            "bbox": (100, 100, 200, 200),
            "confidence": 0.9,
        },
        {
            "label": "parcel",
            "bbox": (400, 400, 500, 500),
            "confidence": 0.85,
        },
    ]

    second_frame = [
        {
            "label": "parcel",
            "bbox": (410, 405, 510, 505),
            "confidence": 0.83,
        },
        {
            "label": "parcel",
            "bbox": (110, 105, 210, 205),
            "confidence": 0.88,
        },
    ]

    first_tracked = tracker.update(first_frame)
    second_tracked = tracker.update(second_frame)

    assert first_tracked[0]["track_id"] == 1
    assert first_tracked[1]["track_id"] == 2

    # Detection order changed, but object identity should not.
    assert second_tracked[0]["track_id"] == 2
    assert second_tracked[1]["track_id"] == 1

def test_tracker_does_not_match_different_object_classes():
    tracker = SimpleTracker()

    first_frame = [
        {
            "label": "parcel",
            "bbox": (100, 100, 200, 200),
            "confidence": 0.9,
        }
    ]

    second_frame = [
        {
            "label": "person",
            "bbox": (105, 105, 205, 205),
            "confidence": 0.88,
        }
    ]

    first_tracked = tracker.update(first_frame)
    second_tracked = tracker.update(second_frame)

    assert first_tracked[0]["track_id"] == 1
    assert second_tracked[0]["track_id"] == 2

def test_tracker_preserves_id_after_temporary_disappearance():
    tracker = SimpleTracker()

    first_frame = [
        {
            "label": "parcel",
            "bbox": (100, 100, 200, 200),
            "confidence": 0.9,
        }
    ]

    second_frame = []

    third_frame = [
        {
            "label": "parcel",
            "bbox": (105, 105, 205, 205),
            "confidence": 0.88,
        }
    ]

    first_tracked = tracker.update(first_frame)
    second_tracked = tracker.update(second_frame)
    third_tracked = tracker.update(third_frame)

    assert first_tracked[0]["track_id"] == 1
    assert second_tracked == []
    assert third_tracked[0]["track_id"] == 1

def test_tracker_creates_new_id_after_track_expires():
    tracker = SimpleTracker(
        max_missing_frames=1
    )

    first_frame = [
        {
            "label": "parcel",
            "bbox": (100, 100, 200, 200),
            "confidence": 0.9,
        }
    ]

    # One missed frame is allowed.
    second_frame = []

    # Track has now been missing too long.
    third_frame = []

    fourth_frame = [
        {
            "label": "parcel",
            "bbox": (105, 105, 205, 205),
            "confidence": 0.88,
        }
    ]

    first_tracked = tracker.update(first_frame)
    second_tracked = tracker.update(second_frame)
    third_tracked = tracker.update(third_frame)
    fourth_tracked = tracker.update(fourth_frame)

    assert first_tracked[0]["track_id"] == 1
    assert second_tracked == []
    assert third_tracked == []

    assert fourth_tracked[0]["track_id"] == 2