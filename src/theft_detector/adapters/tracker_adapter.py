from collections.abc import Iterable, Mapping

from theft_detector.models.detection import Detection


class TrackerAdapter:

    def to_detection(self, tracker_output: Mapping) -> Detection:
        return Detection(
            track_id=tracker_output["track_id"],
            label=tracker_output["label"],
            bbox=tuple(tracker_output["bbox"]),
            confidence=tracker_output.get("confidence", 1.0),
        )

    def to_detections(
        self,
        tracker_outputs: Iterable[Mapping],
    ) -> list[Detection]:
        return [
            self.to_detection(tracker_output)
            for tracker_output in tracker_outputs
        ]