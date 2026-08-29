import streamlit as st
import tempfile
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()
import os
from deepface import DeepFace

# Streamlit UI Configuration
st.set_page_config(page_title="Gender Identifier", page_icon="🧑‍🦰", layout="wide")

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Glassmorphism background for main container */
    .block-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 3rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-top: 2rem;
    }

    /* Title Styling */
    h1 {
        color: #4ECDC4 !important;
        text-align: center;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 2rem !important;
    }

    /* Subheaders */
    h3 {
        color: #A0AEC0 !important;
        font-weight: 600;
    }

    /* Custom Button */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: #4ECDC4;
        color: #1A202C;
        border: none;
        padding: 0.6rem;
        font-weight: 700;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        color: white;
        border: none;
    }

    /* File Uploader styling */
    .stFileUploader {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 15px;
        padding: 1rem;
        border: 1px dashed rgba(255, 255, 255, 0.2);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

st.title("Gender Identifier")

def classify_gender(image_path):
    try:
        # DeepFace analyze returns a list of dictionaries (one for each face detected)
        # We use 'retinaface' backend which is far superior for dark/angled/CGI faces
        result = DeepFace.analyze(
            img_path=image_path, 
            actions=['gender'], 
            enforce_detection=True,
            detector_backend='retinaface'
        )
        
        if isinstance(result, list):
            if len(result) > 1:
                return None, "Multiple faces detected. Please upload an image with exactly one clearly visible person."
            result = result[0]
            
        gender_dict = result['gender']
        # DeepFace returns percentages for 'Man' and 'Woman'
        dominant_gender = max(gender_dict, key=gender_dict.get)
        
        display_gender = "Male" if dominant_gender == "Man" else "Female"
        
        return display_gender, None
    except ValueError as e:
        if "Face could not be detected" in str(e):
            return None, "No clear face detected. Please use a closer, clearer portrait photo."
        return None, str(e)
    except Exception as e:
        return None, str(e)

# Session state init
if "use_camera" not in st.session_state:
    st.session_state.use_camera = False
if "image_captured" not in st.session_state:
    st.session_state.image_captured = False
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "temp_path" not in st.session_state:
    st.session_state.temp_path = ""

# Layout with tabs
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Capture from Webcam"])

with tab1:
    st.markdown("### Upload a Photo")
    uploaded_file = st.file_uploader("Choose a clear portrait image...", type=["png", "jpg", "jpeg", "webp", "heic", "heif", "bmp", "tiff"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp_path = temp.name

        # Convert image to a standard RGB JPEG so OpenCV/DeepFace can read it
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(temp_path, format="JPEG")

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            with st.spinner("Analyzing facial features..."):
                label, error = classify_gender(temp_path)
                
                if label:
                    st.success("✅ Analysis Complete")
                    st.metric(label="Predicted Gender", value=label.capitalize())
                else:
                    st.error(f"❌ Error during analysis: {error}")

with tab2:
    st.markdown("### Take a Photo")
    
    if not st.session_state.get("use_camera", False) and not st.session_state.image_captured:
        if st.button("📸 Enable Camera"):
            st.session_state.use_camera = True
            st.session_state.predicted = False
            st.rerun()

    if st.session_state.get("use_camera", False) and not st.session_state.image_captured:
        camera_image = st.camera_input("Look straight into the camera")
        if camera_image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                st.session_state.temp_path = temp.name
            
            # Convert camera image to a standard RGB JPEG
            image = Image.open(camera_image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(st.session_state.temp_path, format="JPEG")
            
            st.session_state.image_captured = True
            st.session_state.predicted = False
            st.session_state.use_camera = False
            st.rerun()

    if st.session_state.image_captured:
        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open(st.session_state.temp_path), caption="Captured Image", use_container_width=True)

        with col2:
            if not st.session_state.predicted:
                with st.spinner("Analyzing facial features..."):
                    label, error = classify_gender(st.session_state.temp_path)
                    
                    if label:
                        st.success("✅ Analysis Complete")
                        st.metric(label="Predicted Gender", value=label.capitalize())
                    else:
                        st.error(f"❌ Error during analysis: {error}")
                st.session_state.predicted = True

        if st.button("🔄 Retake Image"):
            for key in ("image_captured", "predicted", "temp_path", "use_camera"):
                st.session_state[key] = False
            st.rerun()
