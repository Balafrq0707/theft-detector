import cv2
from ultralytics import YOLO

from theft_detector.detector import TheftDetector
from theft_detector.pipeline import TheftDetectionPipeline
from theft_detector.runtime import TheftDetectionRuntime


VIDEO_PATH = "video.mp4"
MODEL_PATH = "yolov8n.pt"


def main():

    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Create theft detection system
    detector = TheftDetector()

    pipeline = TheftDetectionPipeline(
        detector=detector
    )

    runtime = TheftDetectionRuntime(
        model=model,
        pipeline=pipeline,
    )

    # Open video
    capture = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    frame_index = 0

    while True:

        success, frame = capture.read()

        if not success:
            break

        # Calculate timestamp from video position
        timestamp = frame_index / fps

        # Run the full theft detection pipeline
        events = runtime.process_frame(
            frame=frame,
            timestamp=timestamp,
        )

        # Print detected events
        for event in events:

            print(
                f"\nTHEFT EVENT DETECTED"
            )

            print(
                f"Parcel ID: {event.parcel_id}"
            )

            print(
                f"Person ID: {event.person_id}"
            )

            print(
                f"Timestamp: {event.timestamp}"
            )

            print(
                f"Risk Score: {event.risk_score}"
            )

            print(
                f"Reason: {event.reason}"
            )

        frame_index += 1

    capture.release()


if __name__ == "__main__":

    main()