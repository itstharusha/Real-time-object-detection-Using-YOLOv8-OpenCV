import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# ==================== CUSTOM CSS (Enterprise-grade Design) ====================
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main container adjustments */
    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Hide Streamlit's default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Header styling */
    .main-header {
        text-align: center;
        margin-bottom: 3rem;
    }
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    .main-header p {
        font-size: 1.25rem;
        color: #64748b;
        margin-top: 0.75rem;
        font-weight: 500;
    }

    /* Glassmorphic cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        padding: 2rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
    }

    /* Control panel styling */
    .control-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    /* Custom slider styling */
    .stSlider > label {
        font-weight: 600;
        color: #334155;
    }
    section[data-testid="stSlider"] .css-1g0d3z1 {
        background: linear-gradient(to right, #6366f1, #8b5cf6);
    }

    /* Status indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        background: #f0fdf4;
        color: #166534;
        border: 1px solid #86efac;
    }
    .status-badge.warning {
        background: #fffbeb;
        color: #92400e;
        border-color: #fcd34d;
    }

    /* Webcam container */
    .video-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        position: relative;
    }
    .video-overlay {
        position: absolute;
        top: 1rem;
        left: 1rem;
        z-index: 10;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 5rem;
        padding: 2rem 0;
        color: #94a3b8;
        font-size: 0.875rem;
        border-top: 1px solid #e2e8f0;
    }
    .footer a {
        color: #6366f1;
        text-decoration: none;
        font-weight: 500;
    }
    .footer a:hover {
        text-decoration: underline;
    }

    /* Loading spinner override */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>YOLOv8 Object Detection</h1>
    <p>Real-time, high-performance detection powered by Ultralytics</p>
</div>
""", unsafe_allow_html=True)

# ==================== MODEL LOADING WITH FEEDBACK ====================
@st.cache_resource(show_spinner=False)
def load_model():
    with st.spinner("Loading YOLOv8 model... This may take a moment."):
        return YOLO("yolov8n.pt")

model = load_model()

# Status indicator
st.markdown("""
<div class="status-badge">
    <span>●</span> Model loaded: YOLOv8n (nano)
</div>
""", unsafe_allow_html=True)

# ==================== LAYOUT: TWO COLUMNS ====================
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    # Controls Card
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='control-header'>⚙️ Detection Controls</div>", unsafe_allow_html=True)

    conf = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Lower values show more detections (including weaker ones)"
    )

    iou = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Higher values reduce overlapping bounding boxes"
    )

    # Model info expander
    with st.expander("📊 Model & Performance Info", expanded=False):
        st.markdown("""
        **Model**: YOLOv8n (nano) – Optimized for speed  
        **Inference**: Runs entirely in browser via WebRTC  
        **Expected FPS**: 20–35 on modern hardware  
        **Tip**: For higher accuracy, replace `yolov8n.pt` with `yolov8m.pt` or `yolov8l.pt`
        """)

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    # Video Feed Card
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='control-header'>📹 Live Detection Feed</div>", unsafe_allow_html=True)

    st.caption("Click **START** below and allow camera access. Detection runs in real-time.")

    # Detector class with current parameters
    class YOLODetector:
        def __init__(self, model, conf, iou):
            self.model = model
            self.conf = conf
            self.iou = iou

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            # Run inference
            results = self.model.predict(
                source=img,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
                device="cpu"
            )

            # Annotate frame
            annotated_frame = results[0].plot(
                line_width=2,
                font_size=1,
                labels=True,
                boxes=True,
                masks=False,
                probs=True
            )

            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # WebRTC Streamer
    ctx = webrtc_streamer(
        key="yolov8-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: YOLODetector(model, conf, iou),
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
            "audio": False
        },
        async_processing=True,
        translations={
            "start": "Start Camera",
            "stop": "Stop Camera",
            "select_device": "Select Camera"
        }
    )

    # Live status
    if ctx.state.playing:
        st.success("● Live detection active")
    elif ctx.state.paused:
        st.warning("● Stream paused")
    else:
        st.info("● Click 'Start Camera' to begin detection")

    st.markdown("</div>", unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    Built with <strong>Streamlit</strong> • Powered by <strong>Ultralytics YOLOv8</strong><br>
    <a href="https://github.com/ultralytics/ultralytics" target="_blank">Ultralytics GitHub</a> • 
    <a href="https://docs.ultralytics.com" target="_blank">Documentation</a>
</div>
""", unsafe_allow_html=True)
