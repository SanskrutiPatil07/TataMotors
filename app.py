import streamlit as st
import cv2
import numpy as np

from src.safety_glasses_detector import SafetyGlassesDetector

st.title("🦺 AI Safety Glasses Detector")

st.write("Upload an image to check whether safety glasses are detected.")

image = st.camera_input("Take a picture")

if image is not None:

    file_bytes = np.asarray(
    bytearray(image.read()),
    dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    detector = SafetyGlassesDetector()
    result = detector.detect(image)

    annotated = detector.annotate(image, result)

    st.image(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        caption="Detection Result",
        use_container_width=True
    )

    st.subheader("Result")

    st.write(f"**Label:** {result.label}")
    st.write(f"**Confidence:** {result.confidence:.2f}")
    st.write(f"**Faces Detected:** {result.face_count}")