from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class DetectionResult:
    label: str
    confidence: float
    face_count: int


def _train_model() -> LogisticRegression:
    X = np.array(
        [
            [0.15, 0.03, 0.02, 0.04, 0.03],
            [0.20, 0.08, 0.11, 0.12, 0.10],
            [0.85, 0.35, 0.45, 0.52, 0.41],
            [0.78, 0.30, 0.38, 0.48, 0.37],
            [0.90, 0.40, 0.50, 0.58, 0.45],
            [0.10, 0.02, 0.01, 0.02, 0.02],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1, 1, 0], dtype=np.int64)
    model = LogisticRegression(max_iter=2000, random_state=7)
    model.fit(X, y)
    return model


_MODEL = _train_model()


def extract_features(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.astype(np.uint8)
    gray = cv2.resize(gray, (80, 80), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32)

    mean_intensity = gray.mean() / 255.0
    contrast = gray.std() / 255.0

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_density = np.mean(np.abs(sobel_x) + np.abs(sobel_y) > 25)
    horizontal_strength = np.mean(np.abs(sobel_x)) / 255.0
    vertical_strength = np.mean(np.abs(sobel_y)) / 255.0

    return np.array([mean_intensity, contrast, edge_density, horizontal_strength, vertical_strength], dtype=np.float32)


def predict_safety_glasses(frame: np.ndarray) -> float:
    features = extract_features(frame)
    probability = _MODEL.predict_proba(features.reshape(1, -1))[0, 1]
    return float(probability)


class SafetyGlassesDetector:
    def __init__(self) -> None:
        cascade_dir = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, "haarcascade_frontalface_default.xml"))
        self.eye_glasses_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, "haarcascade_eye_tree_eyeglasses.xml"))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, "haarcascade_eye.xml"))

    def detect(self, image: np.ndarray) -> DetectionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return DetectionResult(label="no_face_detected", confidence=0.0, face_count=0)

        scores: list[float] = []
        for x, y, w, h in faces:
            face_region = image[y : y + h, x : x + w]
            eye_glasses = self.eye_glasses_cascade.detectMultiScale(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=3)
            eyes = self.eye_cascade.detectMultiScale(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=3)
            score = predict_safety_glasses(face_region)
            if len(eye_glasses) > 0:
                score += 0.25
            elif len(eyes) > 0:
                score -= 0.08
            scores.append(min(max(score, 0.0), 1.0))

        confidence = float(np.mean(scores)) if scores else 0.0
        label = "safety_glasses_detected" if confidence >= 0.55 else "no_safety_glasses"
        return DetectionResult(label=label, confidence=confidence, face_count=len(faces))

    def annotate(self, image: np.ndarray, result: DetectionResult) -> np.ndarray:
        annotated = image.copy()
        gray = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        for x, y, w, h in faces:
            color = (0, 255, 0) if result.label == "safety_glasses_detected" else (0, 0, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, f"{result.label}: {result.confidence:.2f}", (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return annotated


def run_detection(image_path: Optional[str] = None, output_path: Optional[str] = None) -> Tuple[str, float, int]:
    if image_path is None:
        raise ValueError("An image path is required for this prototype.")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detector = SafetyGlassesDetector()
    result = detector.detect(image)
    if output_path:
        annotated = detector.annotate(image, result)
        cv2.imwrite(output_path, annotated)

    return result.label, result.confidence, result.face_count


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Detect safety glasses in an image")
    parser.add_argument("--image", required=True, help="Path to an input image")
    parser.add_argument("--output", help="Optional output path for the annotated image")
    args = parser.parse_args()

    label, confidence, face_count = run_detection(args.image, args.output)
    print(json.dumps({"label": label, "confidence": round(confidence, 3), "faces": face_count}))


if __name__ == "__main__":
    main()
