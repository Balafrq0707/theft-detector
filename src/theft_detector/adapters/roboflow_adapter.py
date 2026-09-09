from collections.abc import Mapping


class RoboflowAdapter:

    def to_detections(
        self,
        result: Mapping,
    ) -> list[dict]:

        predictions = (
            result["predictions"]["predictions"]
        )

        detections = []

        for prediction in predictions:

            x = prediction["x"]
            y = prediction["y"]
            width = prediction["width"]
            height = prediction["height"]

            x1 = x - (width / 2)
            y1 = y - (height / 2)
            x2 = x + (width / 2)
            y2 = y + (height / 2)

            detections.append(
                {
                    "label": prediction["class"],
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    "confidence": prediction[
                        "confidence"
                    ],
                }
            )

        return detections