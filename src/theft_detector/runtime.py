from theft_detector.adapters.ultralytics_adapter import (
    UltralyticsAdapter,
)
from theft_detector.models.event import TheftEvent
from theft_detector.pipeline import TheftDetectionPipeline


class TheftDetectionRuntime:

    def __init__(
        self,
        model,
        pipeline: TheftDetectionPipeline,
        ultralytics_adapter: UltralyticsAdapter | None = None,
    ):

        self.model = model
        self.pipeline = pipeline

        self.ultralytics_adapter = (
            ultralytics_adapter
            or UltralyticsAdapter()
        )

    def process_frame(
        self,
        frame,
        timestamp: float,
    ) -> list[TheftEvent]:

        results = self.model.track(
            frame,
            persist=True,
        )

        events: list[TheftEvent] = []

        for result in results:

            tracker_outputs = (
                self.ultralytics_adapter
                .result_to_tracker_outputs(
                    result
                )
            )

            result_events = (
                self.pipeline.process(
                    tracker_outputs=tracker_outputs,
                    timestamp=timestamp,
                )
            )

            events.extend(result_events)

        return events