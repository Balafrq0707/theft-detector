from theft_detector.detector import TheftDetector
from theft_detector.pipeline import TheftDetectionPipeline
from theft_detector.runtime import TheftDetectionRuntime


def test_runtime_processes_model_result():

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

    class MockModel:

        def track(
            self,
            frame,
            persist=True,
        ):
            return [
                MockResult()
            ]

    detector = TheftDetector()

    pipeline = TheftDetectionPipeline(
        detector=detector
    )

    runtime = TheftDetectionRuntime(
        model=MockModel(),
        pipeline=pipeline,
    )

    events = runtime.process_frame(
        frame="fake_frame",
        timestamp=0.0,
    )

    assert events == []

    assert 1 in detector.registry.persons
    assert 10 in detector.registry.parcels