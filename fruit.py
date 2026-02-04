import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# Load your trained YOLOv8 model
model = YOLO("weights/best.pt")

st.title("🍎🍌🍊 Fruit Detector (YOLOv8)")
st.write("Upload an image and let the model detect apples, bananas, and oranges.")

# Upload a single image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    results = model.predict(image, device="cpu")
    result_img = Image.fromarray(results[0].plot()[:, :, ::-1])  # BGR → RGB
    st.image(result_img, caption="Detected Fruits", use_container_width=True)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        st.write(f"**{model.names[cls_id]}** - Confidence: {conf:.2f}")