import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av

# Professional page config
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, modern custom CSS inspired by professional dashboards (rounded cards, subtle shadows, soft colors)
st.markdown("""
<style>
    /* Global padding and font */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    h1 {
        font-weight: 700;
        color: #1e293b;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    h3 {
        color: #475569;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card containers for metrics/controls */
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
    }
    
    /* Metric-like controls section */
    .control-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
    }
    
    /* Slider enhancements */
    section[data-testid="stSlider"] > div > div > div > div {
        background: #e2e8f0;
    }
    
    /* Webcam container – larger, centered, with subtle border */
    .streamlit-webrtc video {
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #64748b;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# Header – clean and centered
st.markdown("<h1>🔍 YOLOv8 Real-Time Object Detection</h1>", unsafe_allow_html=True)
st.markdown("<h3>High-performance detection in your browser webcam</h3>", unsafe_allow_html=True)

# Model loading
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Controls in a professional card layout
st.markdown("<div class='card control-card'>", unsafe_allow_html=True)
st.markdown("#### Detection Settings")

col1, col2 = st.columns(2)
with col1:
    conf = st.slider(
        "Confidence Threshold",
        min_value=0.0, max_value=1.0, value=0.25, step=0.05,
        help="Filter out weak detections"
    )
with col2:
    iou = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.0, max_value=1.0, value=0.45, step=0.05,
        help="Reduce overlapping boxes"
    )
st.markdown("</div>", unsafe_allow_html=True)

# Info card
with st.expander("ℹ️ Model & Performance Details", expanded=False):
    st.markdown("""
    - **Model**: YOLOv8n (nano) – Lightweight and fast for real-time CPU inference  
    - **Mode**: Pure detection (optimized for reliable deployment)  
    - **Expected FPS**: 20–35 on Streamlit Cloud free tier  
    - **Upgrade tip**: Replace `yolov8n.pt` with `yolov8s.pt` or `yolov8m.pt` for higher accuracy
    """)

# Webcam section – prominent and card-wrapped
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("#### 📹 Live Webcam Feed")
st.caption("Click START and allow camera access – detection runs instantly in your browser.")

class Detector:
    def __init__(self):
        self.model = model
        self.conf = conf
        self.iou = iou

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(img, conf=self.conf, iou=self.iou, verbose=False)
        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

webrtc_streamer(
    key="detector",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=Detector,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
st.markdown("</div>", unsafe_allow_html=True)

# Subtle footer
st.markdown("""
<div class='footer'>
    Built with Streamlit • Ultralytics YOLOv8 • 
    <a href='https://github.com/itstharusha/Real-time-object-detection-Using-YOLOv8-OpenCV' target='_blank'>View on GitHub</a>
</div>
""", unsafe_allow_html=True)
