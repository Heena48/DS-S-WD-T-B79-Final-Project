# Fruit Object Detection using Deep Learning

## Project Overview
This project implements **fruit object detection** (banana, orange, apple) using **deep learning models** such as **YOLOv8, Faster R-CNN, and SSD**. The solution is deployed via a **Streamlit application hosted on AWS/GCP**, enabling real-time fruit detection from user-uploaded images.

---

## Problem Statement
Detect and localize fruits in images by:
- Drawing bounding boxes around each fruit.
- Labeling detected fruits correctly.
- Ensuring robustness to variations in lighting, orientation, and occlusion.

---

##  Business Use Cases
- **Smart Retail**: Automated fruit recognition for billing and stock monitoring.  
- **Agriculture**: Yield estimation by detecting fruits on trees.  
- **Food Industry**: Automated fruit sorting on conveyor belts.  
- **Health Tech**: Calorie-tracking apps identifying fruits in meal images.  

---

## Approach
1. **Data Collection**: 240 training images, 60 testing images (banana, orange, apple).  
2. **Annotation**: Bounding boxes via **LabelImg/Roboflow**.  
3. **Preprocessing**: Resize (416x416 or 640x640), normalize, augment (rotation, flips, brightness, noise).  
4. **Model Training**: Transfer learning with YOLOv8/Faster R-CNN (COCO pre-trained).  
5. **Evaluation**: Precision, Recall, F1 Score, mAP.  
6. **Visualization**: Bounding boxes with confidence scores.  
7. **Deployment**: Streamlit app hosted on GCP.  

---

## 📊 Results
- **mAP@0.5 = 0.93**  
- **F1-score = 0.90**  
- Robust detection across lighting and orientation variations.  

---




