import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2

# Page config for a professional touch
st.set_page_config(
    page_title="YOLOv8 Real-Time Object Detection",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, modern, professional look (subtle but effective)
st.markdown("""
    <style>
    .main > div {padding-top: 2rem; padding-bottom: 2rem;}
    .stSlider > div > div > div > div {background-color: #f0f2f6;}
    .css-1y0tuds {font-size: 1.1rem;}
    h1 {font-weight: 600; color: #1e3a8a;}
    .st-bb {border-bottom: 1px solid #e0e0e0;}
    .st-emotion-cache-1kyx0lz {padding: 1rem;}
    </style>
    """, unsafe_allow_html=True)

# Header with markdown for better typography
st.markdown("""
    # 🔍 Real-Time Object Detection  
    ### Powered by **YOLOv8** (Ultralytics)
    Experience state-of-the-art object detection and tracking directly in your browser using your webcam.
    """)

# Load model (cached)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Use yolov8s.pt or custom for better accuracy

model = load_model()

# Info expander (professional way to hide secondary details)
with st.expander("ℹ️ Model & Performance Information", expanded=False):
    st.markdown("""
        - **Model**: YOLOv8n (nano) – Fast & lightweight for smooth real-time performance on CPU.
        - **Tracking**: ByteTrack enabled for persistent object IDs across frames.
        - **Deployment Tip**: On Streamlit Cloud (CPU), expect 15–30 FPS. For higher FPS, deploy on GPU (e.g., Hugging Face Spaces Pro).
        - Swap model: Change `"yolov8n.pt"` to `yolov8s.pt`, `yolov8m.pt`, or your custom `.pt`.
        """)

# Two-column layout for controls (cleaner than plain sidebar)
col1, col2 = st.columns([1, 1])

with col1:
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Lower values detect more objects (may increase false positives)."
    )

with col2:
    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Controls overlap suppression during non-max suppression."
    )

# Advanced settings in an expander (keeps main UI uncluttered)
with st.expander("⚙️ Advanced Settings", expanded=False):
    st.info("These are fixed for optimal real-time performance. Edit code to customize.")
    st.code("""
tracker="bytetrack.yaml"  # Persistent tracking with IDs
persist=True
async_processing=True    # Essential for smooth FPS
    """)

# Processor class (unchanged – core functionality intact)
class YOLOProcessor:
    def __init__(self):
        self.model = model
        self.conf = conf_threshold
        self.iou = iou_threshold

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        results = self.model.track(
            img,
            conf=self.conf,
            iou=self.iou,
            persist=True,
            tracker="bytetrack.yaml"
        )
        
        annotated = results[0].plot()
        
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# Main webcam streamer (centered and prominent)
st.markdown("### 📹 Live Webcam Feed")
st.markdown("> Allow camera access when prompted. Detection runs in real-time.")

webrtc_streamer(
    key="yolo-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YOLOProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# Footer
st.markdown("""
    ---
    Built with ❤️ using **Streamlit**, **streamlit-webrtc**, and **Ultralytics YOLOv8**.  
    Source code: [GitHub Repo](https://github.com/itstharusha/Real-time-object-detection-Using-YOLOv8-OpenCV)
    """)
