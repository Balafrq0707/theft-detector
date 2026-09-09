import math


class SimpleTracker:

    def __init__(
        self,
        distance_threshold: float = 50.0,
        max_missing_frames: int = 1,
    ):
        self.next_track_id = 1
        self.distance_threshold = distance_threshold
        self.max_missing_frames = max_missing_frames

        self.tracks: dict[int, dict] = {}

    def update(
        self,
        detections: list[dict],
    ) -> list[dict]:

        # Increase the missing-frame count for
        # every existing track.
        for track in self.tracks.values():
            track["missing_frames"] += 1

        tracked_detections = []

        matched_track_ids = set()

        for detection in detections:

            center = self._center(
                detection["bbox"]
            )

            matched_track_id = (
                self._find_matching_track(
                    label=detection["label"],
                    center=center,
                    matched_track_ids=matched_track_ids,
                )
            )

            if matched_track_id is None:

                track_id = self.next_track_id
                self.next_track_id += 1

            else:

                track_id = matched_track_id
                matched_track_ids.add(track_id)

            self.tracks[track_id] = {
                "label": detection["label"],
                "center": center,
                "missing_frames": 0,
            }

            tracked_detections.append(
                {
                    **detection,
                    "track_id": track_id,
                }
            )

        # Remove tracks that have been missing
        # for too many frames.
        expired_track_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missing_frames"]
            > self.max_missing_frames
        ]

        for track_id in expired_track_ids:
            del self.tracks[track_id]

        return tracked_detections

    def _find_matching_track(
        self,
        label: str,
        center: tuple[float, float],
        matched_track_ids: set[int],
    ) -> int | None:

        closest_track_id = None
        closest_distance = float("inf")

        for track_id, track in self.tracks.items():

            if track_id in matched_track_ids:
                continue

            if track["label"] != label:
                continue

            distance = math.dist(
                center,
                track["center"],
            )

            if (
                distance <= self.distance_threshold
                and distance < closest_distance
            ):
                closest_distance = distance
                closest_track_id = track_id

        return closest_track_id

    @staticmethod
    def _center(
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float]:

        x1, y1, x2, y2 = bbox

        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
        )