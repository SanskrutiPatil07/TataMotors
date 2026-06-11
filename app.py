import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import tempfile

st.title("🦺 AI Safety Glasses Detector")

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=st.secrets["ROBOFLOW_API_KEY"]
)

image = st.camera_input("Take a picture")

if image is not None:

    img = Image.open(image)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name)

        result = CLIENT.infer(
            tmp.name,
            model_id="safety-glasses-detection-qkhel/1"
        )

    predictions = result.get("predictions", [])

    st.image(img, caption="Captured Image")

    if len(predictions) > 0:
        st.success("🟢 Safety Glasses Detected")
        st.write(predictions)
    else:
        st.error("🔴 No Safety Glasses Detected")