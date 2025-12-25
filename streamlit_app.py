import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2
import threading

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
    /* Import Inter – the most widely used dashboard font in 2025 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Responsive container */
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    @media (max-width: 1024px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }

    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Header */
    .main-header {
        text-align: center;
        margin-bottom: 3rem;
    }
    .main-header h1 {
        font-size: 2.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .main-header p {
        font-size: 1.25rem;
        color: #64748b;
        margin-top: 0.75rem;
        font-weight: 500;
    }

    /* Glassmorphic cards with dark mode support */
    .glass-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        padding: 2rem;
        transition: all 0.3s ease;
    }
    @media (prefers-color-scheme: dark) {
        .glass-card {
            background: rgba(30, 30, 30, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    }
    .glass-card:hover {
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
    }

    /* Control header */
    .control-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    /* Video container – now applied */
    .video-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        position: relative;
    }

    /* Status badges */
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

    /* Responsive columns */
    @media (max-width: 768px) {
        div[data-testid="column"] {
            width: 100% !important;
            margin-bottom: 2rem;
        }
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

# ==================== MODEL LOADING ====================
@st.cache_resource(show_spinner=False)
def load_model():
    with st.spinner("Loading YOLOv8 model..."):
        return YOLO("yolov8n.pt")

model = load_model()

st.markdown("""
<div class="status-badge">
    <span>●</span> Model loaded: YOLOv8n (nano)
</div>
""", unsafe_allow_html=True)

# ==================== RESPONSIVE LAYOUT ====================
# Use CSS media query fallback + flexible column ratios
left_col, right_col = st.columns([1, 1.8], gap="large")

with left_col:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='control-header'>⚙️ Detection Controls</div>", unsafe_allow_html=True)

    conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05,
                     help="Lower values show more detections (including weaker ones)")
    iou = st.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.45, 0.05,
                    help="Higher values reduce overlapping bounding boxes")

    with st.expander("📊 Model & Performance Info", expanded=False):
        st.markdown("""
        **Model**: YOLOv8n (nano) – Optimized for speed  
        **Inference**: Real-time via WebRTC  
        **Expected FPS**: 20–35 on modern hardware  
        **Tip**: Replace `yolov8n.pt` with larger models for higher accuracy
        """)

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='control-header'>📹 Live Detection Feed</div>", unsafe_allow_html=True)
    st.caption("Click **Start Camera** and allow access. Parameters update live.")

    # Real-time parameter processor
    class YOLODetector(VideoProcessorBase):
        def __init__(self):
            self.lock = threading.Lock()
            self.conf = conf
            self.iou = iou

        def update_params(self, conf, iou):
            with self.lock:
                self.conf = conf
                self.iou = iou

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            with self.lock:
                current_conf = self.conf
                current_iou = self.iou

            results = model.predict(source=img, conf=current_conf, iou=current_iou,
                                    verbose=False, device="cpu")

            annotated = results[0].plot(line_width=2, font_size=1, labels=True,
                                       boxes=True, probs=True)

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    # Streamer with live parameter updates
    ctx = webrtc_streamer(
        key="yolov8-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLODetector,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False},
        async_processing=True,
        translations={"start": "Start Camera", "stop": "Stop Camera"}
    )

    # Update processor params on slider change
    if ctx.video_processor:
        ctx.video_processor.update_params(conf, iou)

    # Status
    if ctx.state.playing:
        st.success("● Live detection active")
    elif ctx.state.paused:
        st.warning("● Stream paused")
    else:
        st.info("● Ready – Click 'Start Camera'")

    # Apply video container styling
    st.markdown("<div class='video-container'>", unsafe_allow_html=True)
    # The webrtc component renders inside the previous container automatically
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    Built with <strong>Streamlit</strong> • Powered by <strong>Ultralytics YOLOv8</strong><br>
    <a href="https://ultralytics.com" target="_blank">Ultralytics</a> • 
    <a href="https://docs.ultralytics.com" target="_blank">Documentation</a>
</div>
""", unsafe_allow_html=True)
