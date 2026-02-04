import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# Load your trained YOLOv8 model
model = YOLO("C:/Users/Windows/runs/detect/train7/weights/best.pt")

st.title("🍎🍌🍊 Fruit Detector (YOLOv8)")
st.write("Upload an image or run detection on the entire test dataset.")

# Option 1: Upload a single image
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
        st.write(f"Class: {model.names[cls_id]}, Confidence: {conf:.2f}")

# Option 2: Run on entire test dataset
test_folder = "C:/Users/Windows/Downloads/Data-final_pro/Data/Test"

if st.button("Run detection on all test images"):
    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for img_file in image_files:
        img_path = os.path.join(test_folder, img_file)
        results = model.predict(img_path, device="cpu")

        result_img = Image.fromarray(results[0].plot()[:, :, ::-1])  # BGR → RGB
        st.image(result_img, caption=f"Detected Fruits in {img_file}", use_container_width=True)

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"Class: {model.names[cls_id]}, Confidence: {conf:.2f}")