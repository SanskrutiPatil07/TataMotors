# Tata Motors Safety Glasses Detection

This repository contains a lightweight prototype for detecting whether a person is wearing safety glasses using computer vision and a small trained classifier.

## What it does
- Detects faces in an image or webcam stream.
- Looks for eye regions and glass-like structures.
- Uses a compact machine-learning classifier to estimate whether safety glasses are present.

## Installation
```bash
cd /workspaces/TataMotors
pip install -r requirements.txt
```

## Usage
Run the detector on a single image:
```bash
python app.py --image /path/to/person.jpg --output /tmp/annotated.jpg
```

Run the detector on a live camera stream:
```bash
python app.py --camera
```

## Project structure
- app.py: command-line entry point
- src/safety_glasses_detector.py: detector logic and classifier
- tests/test_detector.py: regression tests

## Notes
This is a prototype intended for demonstration and further training with a real labeled dataset. Accuracy improves significantly with a custom dataset and a deeper model such as YOLOv8 or MediaPipe-based object detection.
