import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
import joblib
import cv2
from PIL import ImageEnhance

def contains_face(image_path):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
    return len(faces) > 0

# Setup base directory
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "face_cnn_model.pth")
encoder_path = os.path.join(BASE_DIR, "label_encoder.joblib")

# Load model
num_classes = 2 
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# Load Label Encoder
label_encoder = joblib.load(encoder_path)
class_names = label_encoder.classes_.tolist()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.65

# GUI Setup
root = tk.Tk()
root.title("🧑‍🦰 Gender Identifier")
root.geometry("600x500")
root.configure(bg="#f4f4f4")

title = tk.Label(root, text="🧑‍🦰 Gender Identifier", font=("Helvetica", 18, "bold"), bg="#f4f4f4", fg="#333")
title.pack(pady=20)

img_panel = tk.Label(root, bg="#f4f4f4")
img_panel.pack()

result_label = tk.Label(root, text="", font=("Arial", 14), bg="#f4f4f4", fg="blue")
result_label.pack(pady=20)

def upload_and_classify():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    try:
        if not contains_face(file_path):
            result_label.config(
                text="❌ No human face detected.\nPlease upload a valid image.",
                fg="red"
            )
            return

        img = Image.open(file_path).convert("RGB")
        if img.width < 100 or img.height < 100:
            base_width = 100
            w_percent = (base_width / float(img.width))
            h_size = int((float(img.height) * float(w_percent)))
            img = img.resize((base_width, h_size), Image.LANCZOS)

        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)  
        img = ImageEnhance.Contrast(img).enhance(1.2)
        img = ImageEnhance.Brightness(img).enhance(1.1) 

        display_img = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(display_img)
        img_panel.config(image=img_tk)
        img_panel.image = img_tk

        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            max_prob, pred_class = torch.max(probabilities, dim=1)
            confidence = max_prob.item()
            label = label_encoder.inverse_transform([pred_class.item()])[0]

        if confidence >= CONFIDENCE_THRESHOLD:
            result_label.config(
                text=f"Predicted: {label}", fg="green"
            )
        else:
            result_label.config(
                text="⚠️ Uncertain prediction.\nPlease upload a clearer image.",
                fg="orange"
            )

    except Exception as e:
        messagebox.showerror("Error", f"Failed to classify image.\n\n{e}")

def capture_and_classify_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam not accessible.")
        return

    # Read frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        messagebox.showerror("Error", "Failed to capture image from webcam.")
        return

    # Save to temp file
    temp_path = "temp_webcam.jpg"
    cv2.imwrite(temp_path, frame)

    # Classify it
    try:
        if not contains_face(temp_path):
            result_label.config(
                text="❌ No human face detected in the captured frame.",
                fg="red"
            )
            os.remove(temp_path)
            return

        img = Image.open(temp_path).convert("RGB")
        os.remove(temp_path)

        if img.width < 100 or img.height < 100:
            base_width = 100
            w_percent = (base_width / float(img.width))
            h_size = int((float(img.height) * float(w_percent)))
            img = img.resize((base_width, h_size), Image.LANCZOS)

        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        img = ImageEnhance.Contrast(img).enhance(1.2)
        img = ImageEnhance.Brightness(img).enhance(1.1)

        display_img = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(display_img)
        img_panel.config(image=img_tk)
        img_panel.image = img_tk

        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            max_prob, pred_class = torch.max(probabilities, dim=1)
            confidence = max_prob.item()
            label = label_encoder.inverse_transform([pred_class.item()])[0]

        if confidence >= CONFIDENCE_THRESHOLD:
            result_label.config(
                text=f"Predicted: {label}", fg="green"
            )
        else:
            result_label.config(
                text="⚠️ Uncertain prediction.\nPlease try with better lighting or framing.",
                fg="orange"
            )

    except Exception as e:
        messagebox.showerror("Error", f"Webcam classification failed.\n\n{e}")

def classify_from_file(file_path):
    try:
        if not contains_face(file_path):
            result_label.config(
                text="❌ No human face detected.\nPlease upload a valid image.",
                fg="red"
            )
            return

        img = Image.open(file_path).convert("RGB")
        if img.width < 100 or img.height < 100:
            base_width = 100
            w_percent = (base_width / float(img.width))
            h_size = int((float(img.height) * float(w_percent)))
            img = img.resize((base_width, h_size), Image.LANCZOS)

        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        img = ImageEnhance.Contrast(img).enhance(1.2)
        img = ImageEnhance.Brightness(img).enhance(1.1)

        display_img = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(display_img)
        img_panel.config(image=img_tk)
        img_panel.image = img_tk

        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            max_prob, pred_class = torch.max(probabilities, dim=1)
            confidence = max_prob.item()
            label = label_encoder.inverse_transform([pred_class.item()])[0]

        if confidence >= CONFIDENCE_THRESHOLD:
            result_label.config(
                text=f"Predicted: {label}", fg="green"
            )
        else:
            result_label.config(
                text="⚠️ Uncertain prediction.\nPlease upload a clearer image.",
                fg="orange"
            )

    except Exception as e:
        messagebox.showerror("Error", f"Classification failed.\n\n{e}")

def preview_webcam():
    preview_window = tk.Toplevel(root)
    preview_window.title("📷 Webcam Preview")
    preview_window.geometry("640x520")
    preview_window.configure(bg="#f4f4f4")

    cam_label = tk.Label(preview_window)
    cam_label.pack()

    cap = cv2.VideoCapture(0)

    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img.resize((640, 480)))
            cam_label.imgtk = imgtk
            cam_label.configure(image=imgtk)
        cam_label.after(10, update_frame)

    def capture_frame():
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image.")
            return
        cap.release()
        preview_window.destroy()

        temp_path = "temp_webcam.jpg"
        cv2.imwrite(temp_path, frame)
        classify_from_file(temp_path)
        os.remove(temp_path)

    def on_close():
        cap.release()
        preview_window.destroy()

    capture_btn = tk.Button(preview_window, text="📸 Capture", command=capture_frame,
                            font=("Arial", 12), bg="#28a745", fg="white")
    capture_btn.pack(pady=10)

    preview_window.protocol("WM_DELETE_WINDOW", on_close)
    update_frame()

webcam_btn = tk.Button(root, text="📷 Use Webcam", command=preview_webcam,
                       font=("Arial", 12), bg="#28a745", fg="white")
webcam_btn.pack(pady=10)

btn = tk.Button(root, text="📁 Upload Image", command=upload_and_classify, font=("Arial", 12),
                bg="#0078D7", fg="white")
btn.pack(pady=10)

if __name__ == "__main__":
    root.mainloop()