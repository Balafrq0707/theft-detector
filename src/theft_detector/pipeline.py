from collections.abc import Iterable, Mapping

from theft_detector.adapters.tracker_adapter import (
    TrackerAdapter,
)
from theft_detector.detector import TheftDetector
from theft_detector.models.event import TheftEvent


class TheftDetectionPipeline:

    def __init__(
        self,
        detector: TheftDetector,
        tracker_adapter: TrackerAdapter | None = None,
    ):

        self.detector = detector

        self.tracker_adapter = (
            tracker_adapter
            or TrackerAdapter()
        )

    def process(
        self,
        tracker_outputs: Iterable[Mapping],
        timestamp: float,
    ) -> list[TheftEvent]:

        detections = (
            self.tracker_adapter.to_detections(
                tracker_outputs
            )
        )

        return self.detector.update(
            detections=detections,
            timestamp=timestamp,
        )