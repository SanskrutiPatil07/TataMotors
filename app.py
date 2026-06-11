import streamlit as st
import requests

st.title("🦺 AI Safety Glasses Detector")

API_KEY = st.secrets["ROBOFLOW_API_KEY"]
MODEL_ID = "safety-glasses-detection-qkhel/1"

image = st.camera_input("Take a picture")

if image is not None:

    image_bytes = image.getvalue()

    st.image(image_bytes, caption="Captured Image")

    response = requests.post(
        f"https://detect.roboflow.com/{MODEL_ID}",
        params={"api_key": API_KEY},
        files={"file": image_bytes}
    )

    result = response.json()

    st.subheader("Detection Result")

    st.write(result)

    if len(result.get("predictions", [])) > 0:
        st.success("🟢 Safety Glasses Detected")
    else:
        st.error("🔴 No Safety Glasses Detected")