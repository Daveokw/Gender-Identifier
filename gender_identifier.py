import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageEnhance
import numpy as np
import joblib
import cv2
import tempfile
import os

def contains_face(image_path):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.05, 3)
    return len(faces) > 0

# Load model
model_path = "face_cnn_model.pth"
encoder_path = "label_encoder.joblib"

num_classes = 2
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

label_encoder = joblib.load(encoder_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

CONFIDENCE_THRESHOLD = 0.65

def classify_image(img: Image.Image):
    img = img.convert("RGB")

    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Brightness(img).enhance(1.1)

    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        max_prob, pred_class = torch.max(probabilities, dim=1)
        confidence = max_prob.item()
        label = label_encoder.inverse_transform([pred_class.item()])[0]

    return label, confidence, img

# Streamlit UI
st.set_page_config(page_title="🧑‍🦰 Gender Identifier", layout="centered")
st.title("🧑‍🦰 Gender Identifier")

# Session state init
if "use_camera" not in st.session_state:
    st.session_state.use_camera = False
if "image_captured" not in st.session_state:
    st.session_state.image_captured = False
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "temp_path" not in st.session_state:
    st.session_state.temp_path = ""

st.subheader("📁 Upload Image")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(uploaded_file.read())
        temp_path = temp.name

    if contains_face(temp_path):
        image = Image.open(temp_path)
        label, confidence, enhanced_img = classify_image(image)

        st.image(enhanced_img, caption="Uploaded Image", use_container_width=True)

        if confidence >= CONFIDENCE_THRESHOLD:
            st.success(f"Predicted Gender: {label}")
        else:
            st.warning("⚠️ Uncertain prediction. Please upload a clearer image.")
    else:
        st.error("❌ No human face detected. Please upload a valid image.")

# Webcam Capture 
st.markdown("---")
st.subheader("📷 Capture from Webcam")
 
if not st.session_state.get("use_camera", False) and not st.session_state.image_captured:
    if st.button("📷 Open Camera"):
        st.session_state.use_camera = True
        st.session_state.predicted = False

if st.session_state.get("use_camera", False) and not st.session_state.image_captured:
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp.write(camera_image.getbuffer())
            st.session_state.temp_path = temp.name
            st.session_state.image_captured = True
            st.session_state.predicted = False
        st.session_state.use_camera = False
        st.rerun()

if st.session_state.image_captured:
    st.image(Image.open(st.session_state.temp_path),
             caption="Captured Image", use_container_width=True)

    if not st.session_state.predicted:
        if contains_face(st.session_state.temp_path):
            label, confidence, _ = classify_image(Image.open(st.session_state.temp_path))
            if confidence >= CONFIDENCE_THRESHOLD:
                st.success(f"Predicted Gender: **{label}**")
            else:
                st.warning("⚠️ Uncertain Prediction.")
        else:
            st.error("❌ No human face detected.")
        st.session_state.predicted = True

    if st.button("Retake Image"):
        for key in ("image_captured", "predicted", "temp_path", "use_camera"):
            st.session_state[key] = False
        st.rerun()
