from theft_detector.adapters.roboflow_adapter import (
    RoboflowAdapter,
)


def test_roboflow_adapter_converts_predictions():
    adapter = RoboflowAdapter()

    result = {
        "predictions": {
            "image": {
                "width": 720,
                "height": 1280,
            },
            "predictions": [
                {
                    "width": 295.0,
                    "height": 287.0,
                    "x": 255.5,
                    "y": 873.5,
                    "confidence": 0.8766,
                    "class_id": 0,
                    "class": "parcel",
                },
                {
                    "width": 157.0,
                    "height": 266.0,
                    "x": 641.5,
                    "y": 518.0,
                    "confidence": 0.7073,
                    "class_id": 1,
                    "class": "person",
                },
            ],
        }
    }

    detections = adapter.to_detections(result)

    assert len(detections) == 2

    assert detections[0]["label"] == "parcel"
    assert detections[0]["confidence"] == 0.8766

    assert detections[1]["label"] == "person"
    assert detections[1]["confidence"] == 0.7073