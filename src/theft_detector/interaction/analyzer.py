from dataclasses import dataclass

from theft_detector.models.detection import Detection
from theft_detector.utils.geometry import distance_between_boxes


@dataclass
class Interaction:

    person_id: int

    parcel_id: int

    distance: float


class InteractionAnalyzer:

    def __init__(
        self,
        interaction_distance: float,
    ):

        self.interaction_distance = interaction_distance

    def find_interactions(
        self,
        persons: list[Detection],
        parcels: list[Detection],
    ) -> list[Interaction]:

        interactions: list[Interaction] = []

        for person in persons:

            for parcel in parcels:

                distance = distance_between_boxes(
                    person.bbox,
                    parcel.bbox,
                )

                if distance <= self.interaction_distance:

                    interactions.append(
                        Interaction(
                            person_id=person.track_id,
                            parcel_id=parcel.track_id,
                            distance=distance,
                        )
                    )

        return interactions