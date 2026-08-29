# Gender Identifier

A smart, modern machine learning app built with Python and Streamlit that instantly and accurately identifies whether a face image is male or female from a single uploaded or captured image.

## Live Demo
Try the live app here: [Gender Identifier on Streamlit](https://gender-identifier.streamlit.app)
## About the Project
Gender Identifier uses the state-of-the-art **DeepFace** library to extract facial features and predict gender. Originally utilizing a locally-trained ResNet18 model, it has been overhauled to leverage industry-standard pre-trained facial recognition weights (such as VGG-Face) to guarantee robust and reliable predictions in various lighting conditions and angles.

### Key Features:
- **Deep Learning**: Utilizes industry-standard models through DeepFace for state-of-the-art facial analysis and gender prediction.
- **Premium Web Interface**: A sleek, interactive, glassmorphism-styled UI built with Streamlit allowing users to upload an image or capture one from a webcam.
- **High Accuracy**: Bypasses the need for local training on small datasets by leveraging massive, pre-trained facial demographic models.
- **Instant Results**: Fast inference delivering prediction and confidence metrics instantly.

## Technology Stack
- **Machine Learning**: DeepFace, TensorFlow/Keras backend
- **Web Framework**: Streamlit
- **Computer Vision**: OpenCV (Headless)
- **Data Processing**: Pillow (PIL), NumPy
- **Language**: Python 3.x

## How to Run Locally

1. **Clone the repository (or download the files):**
```powershell
git clone https://github.com/Daveokw/Gender-Identifier
cd "Gender-Identifier"
```

2. **Create a virtual environment (Recommended):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

3. **Install the required dependencies:**
```powershell
pip install -r requirements.txt
```

4. **Run the Streamlit App:**
```powershell
streamlit run gender_identifier.py
```
*(Note: DeepFace will download the base model weights automatically upon the very first run.)*

> [!NOTE]  
> **Cloud vs. Local TensorFlow:** The `requirements.txt` file is optimized for Streamlit Community Cloud and uses `tensorflow-cpu` to prevent out-of-memory errors during deployment. The performance is **identical** to the standard version on CPU instances. However, if you are running this locally on a machine with a powerful NVIDIA GPU and want hardware acceleration, you can optionally install standard `tensorflow` (the ~500MB version) instead.

## Known Limitations
Due to the nature of AI training datasets, the model may occasionally struggle to accurately classify images when subjects are wearing head coverings, accessories, or other items that significantly obscure natural facial features and hairlines.

## Let's Connect
Feel free to reach out if you have questions, want to contribute, or just want to collaborate on AI projects!
