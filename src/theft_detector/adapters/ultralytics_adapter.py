from collections.abc import Mapping


class UltralyticsAdapter:

    CLASS_NAMES = {
        0: "person",
        1: "parcel",
    }

    def to_tracker_output(
        self,
        box: Mapping,
    ) -> dict:

        return {
            "track_id": box["track_id"],
            "label": box["label"],
            "bbox": tuple(box["bbox"]),
        }

    def result_to_tracker_outputs(
        self,
        result,
    ) -> list[dict]:

        boxes = result.boxes

        # No detections in this result.
        if boxes is None:
            return []

        # Detections may exist before the tracker
        # assigns track IDs.
        if boxes.id is None:
            return []

        outputs = []

        for track_id, class_id, bbox in zip(
            boxes.id,
            boxes.cls,
            boxes.xyxy,
        ):

            label = self.CLASS_NAMES.get(
                int(class_id)
            )

            if label is None:
                continue

            outputs.append(
                {
                    "track_id": int(track_id),
                    "label": label,
                    "bbox": tuple(bbox),
                }
            )

        return outputs