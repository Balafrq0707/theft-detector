from theft_detector.adapters.ultralytics_adapter import (
    UltralyticsAdapter,
)


def test_ultralytics_adapter_handles_no_boxes():

    class MockResult:

        def __init__(self):
            self.boxes = None

    adapter = UltralyticsAdapter()

    outputs = adapter.result_to_tracker_outputs(
        MockResult()
    )

    assert outputs == []