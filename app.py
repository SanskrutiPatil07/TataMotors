import streamlit as st
import requests

st.set_page_config(page_title="AI Safety Glasses Detector")

st.title("🦺 AI Safety Glasses Detector")

API_KEY = st.secrets["ROBOFLOW_API_KEY"]
MODEL_ID = "safety-glasses-detection-qkhel/1"

image = st.camera_input("Take a picture")

if image is not None:

    image_bytes = image.getvalue()

    st.image(image_bytes, caption="Captured Image")

    try:
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

        if "message" in result:
            st.warning(result["message"])

        else:
            predictions = result.get("predictions", [])

            if len(predictions) > 0:

                safety_found = False
                confidence = 0

                for pred in predictions:

                    if pred["class"] == "Safety-Glasses-Detection":
                        safety_found = True
                        confidence = pred["confidence"]
                        break

                if safety_found:

    st.success(
        f"🟢 SAFETY GLASSES DETECTED "
        f"(Confidence: {confidence:.2%})"
    )

    st.success("✅ MACHINE STATUS : ENABLED")

else:

    st.error("🔴 SAFETY GLASSES REQUIRED")

    st.error("❌ MACHINE STATUS : DISABLED")