from io import BytesIO

import pillow_heif
import streamlit as st
from PIL import Image

from gender_analysis import AnalysisOutcome, analyse_gender

pillow_heif.register_heif_opener()

# Streamlit UI Configuration
st.set_page_config(page_title="Gender Identifier", page_icon="🧑‍🦰", layout="wide")

# Custom CSS for Premium Design
st.markdown(
    """
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

    /* Keep Streamlit's implementation details out of the user-facing uploader. */
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        display: none !important;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4ECDC4;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Gender Identifier")
st.caption("Upload or capture a photo containing one clearly visible face.")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
IMAGE_EXTENSIONS = (
    "jpg",
    "jpeg",
    "jpe",
    "jfif",
    "png",
    "webp",
    "heic",
    "heif",
    "bmp",
    "dib",
    "gif",
    "tif",
    "tiff",
)


@st.cache_data(show_spinner=False, max_entries=8)
def analyse_image_bytes(image_bytes: bytes) -> AnalysisOutcome:
    """Decode and analyse one image without leaving temporary files behind."""

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            return analyse_gender(image)
    except (OSError, ValueError):
        return AnalysisOutcome(message="This file is not a readable image.")


def render_analysis(image_file, caption: str) -> None:
    """Render an image and its quality-checked result."""

    image_bytes = image_file.getvalue()
    if not image_bytes:
        st.warning("The image is empty. Please choose another file.")
        return
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        st.warning("The image is too large. Please upload a file smaller than 10 MB.")
        return

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            preview = image.copy()
    except OSError:
        st.warning("This file is not a readable image.")
        return

    image_column, result_column = st.columns(2)
    with image_column:
        st.image(preview, caption=caption, width="stretch")

    with result_column:
        with st.spinner("Analysing facial features..."):
            outcome = analyse_image_bytes(image_bytes)

        if outcome.prediction is None:
            st.warning("No reliable estimate")
            st.info(outcome.message or "Try another image.")
            return

        st.success("Analysis complete")
        st.metric("Prediction", outcome.prediction.label)


if "camera_enabled" not in st.session_state:
    st.session_state.camera_enabled = False


def set_camera_enabled(enabled: bool) -> None:
    """Change camera availability before Streamlit renders the camera widget."""

    st.session_state.camera_enabled = enabled


upload_tab, camera_tab = st.tabs(["Upload image", "Use camera"])

with upload_tab:
    st.markdown("### Upload a photo")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=IMAGE_EXTENSIONS,
    )
    if uploaded_file is not None:
        render_analysis(uploaded_file, "Uploaded image")

with camera_tab:
    st.markdown("### Take a photo")
    if not st.session_state.camera_enabled:
        st.info("Camera is off. Select Enable camera to continue.")
        st.button(
            "Enable camera",
            key="enable-camera",
            type="primary",
            on_click=set_camera_enabled,
            args=(True,),
        )
    else:
        st.button(
            "Turn off camera",
            key="disable-camera",
            on_click=set_camera_enabled,
            args=(False,),
        )
        camera_image = st.camera_input(
            "Keep one face clearly visible and look towards the camera",
            key="camera-input",
            help="Camera permission is controlled by your browser.",
        )
        if camera_image is not None:
            render_analysis(camera_image, "Captured image")

st.caption(
    "This model estimates two presentation categories from facial appearance. It may be "
    "wrong and cannot determine gender identity; do not use it for consequential decisions."
)
