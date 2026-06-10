import streamlit as st
import requests
import cv2
import numpy as np

st.title("🦺 AI Safety Glasses Detector")

st.write("Capture an image and check PPE compliance.")

API_KEY = st.secrets["ROBOFLOW_API_KEY"]

MODEL_ID = "safety-glasses-detection-qkhel/1"

image = st.camera_input("Take a picture")

if image is not None:

    image_bytes = image.getvalue()

    st.image(image_bytes, caption="Captured Image")

    response = requests.post(
        f"https://detect.roboflow.com/{MODEL_ID}",
        params={
            "api_key": API_KEY
        },
        files={
            "file": image_bytes
        }
    )

    result = response.json()

    st.subheader("Detection Result")

    if "predictions" in result and len(result["predictions"]) > 0:

        for prediction in result["predictions"]:

            st.success(
                f"Detected: {prediction['class']} "
                f"(Confidence: {prediction['confidence']:.2f})"
            )

    else:
        st.error("No safety glasses detected.")