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

    predictions = result.get("predictions", [])

    if len(predictions) > 0:

        detected_class = predictions[0]["class"]

        if detected_class == "Safety-Glasses-Detection":
            st.success("🟢 Safety Glasses Detected")

        elif detected_class == "no-Safety-Glasses-Detection":
            st.error("🔴 No Safety Glasses Detected")

        else:
            st.warning(f"Detected: {detected_class}")

    else:
        st.warning("⚠️ No person detected")