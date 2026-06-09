from __future__ import annotations

import argparse

import cv2

from src.safety_glasses_detector import SafetyGlassesDetector, run_detection


def run_camera() -> None:
    detector = SafetyGlassesDetector()
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Unable to access the camera. Please verify that a camera is connected.")

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        result = detector.detect(frame)
        annotated = detector.annotate(frame, result)
        cv2.putText(annotated, f"{result.label} ({result.confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Safety Glasses Detector", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the AI safety glasses detection prototype")
    parser.add_argument("--image", help="Path to an input image")
    parser.add_argument("--output", help="Optional output path for the annotated image")
    parser.add_argument("--camera", action="store_true", help="Run the detector on the local camera stream")
    args = parser.parse_args()

    if args.camera:
        run_camera()
        return

    if not args.image:
        raise SystemExit("Provide --image for a single image run or --camera to stream from a camera.")

    label, confidence, faces = run_detection(args.image, args.output)
    print(f"label={label} confidence={confidence:.3f} faces={faces}")


if __name__ == "__main__":
    main()
