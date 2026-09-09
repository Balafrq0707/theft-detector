from dataclasses import dataclass
from typing import Tuple


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    track_id: int

    label: str

    bbox: BBox

    confidence: float = 1.0

    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox

        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
        )