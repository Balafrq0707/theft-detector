import math

from theft_detector.models.detection import BBox


def center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )


def distance_between_boxes(
    box_a: BBox,
    box_b: BBox,
) -> float:

    ax, ay = center(box_a)
    bx, by = center(box_b)

    return math.sqrt(
        (bx - ax) ** 2 +
        (by - ay) ** 2
    )