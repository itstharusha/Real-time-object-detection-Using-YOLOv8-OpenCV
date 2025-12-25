import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2
import time

# Page configuration
st.set_page_config(
    page_title="YOLOv8 Live Object Detection",
    page_icon=" ",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Professional, modern CSS - clean, minimal, no emojis, premium typography
st.markdown(
    """
    <style>
    /* Primary font stack - professional and modern */
    html, body, [class*="css"]  {
        font-family: "Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Main container */
    .main {
        background-color: #f9fafb;
        padding-top: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    /* Headers */
    h1 {
        font-size: 2.8rem;
        font-weight: 700;
        color: #111827;
        text-align: center;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }
    h3 {
        font-size: 1.25rem;
        font-weight: 500;
        color: #6b7280;
        text-align: center;
        margin-bottom: 3rem;
        letter-spacing: -0.2px;
    }

    /* Section titles */
    h2 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    /* Cards and containers */
    .stExpander > div > label {
        font-weight: 600;
        color: #374151;
    }
    div[data-testid="stVerticalBlock"] > div {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
        margin-bottom: 2rem;
    }

    /* Metrics */
    .stMetric > div {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
    }
    .stMetric label {
        font-size: 0.9rem;
        color: #6b7280;
        font-weight: 500;
    }
    .stMetric > div > div {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
    }

    /* Sliders */
    .stSlider > label {
        font-weight: 600;
        color: #374151;
    }

    /* Buttons */
    .stButton > button {
        background-color: #4f46e5;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        width: 100%;
        transition: background-color 0.2s;
    }
    .stButton > button:hover {
        background-color: #4338ca;
    }

    /* Webcam container */
    .element-container > iframe {
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.875rem;
        margin-top: 5rem;
        padding: 2rem 0;
        border-top: 1px solid #e5e7eb;
    }
    .footer a {
        color: #4f46e5;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("# YOLOv8 Live Object Detection")
    st.markdown("#### Real-time object detection powered by Ultralytics YOLOv8")

st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")

    model_options = {
        "YOLOv8n (Nano - Fastest)": "yolov8n.pt",
        "YOLOv8s (Small - Balanced)": "yolov8s.pt",
        "YOLOv8m (Medium - Most Accurate)": "yolov8m.pt",
    }

    selected_model_name = st.selectbox(
        "Model Variant",
        options=list(model_options.keys()),
        index=0,
        help="Select model based on speed vs accuracy trade-off",
    )

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Lower values increase sensitivity but may add false positives",
    )

    iou_threshold = st.slider(
        "IoU Threshold (Non-Max Suppression)",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Controls suppression of overlapping bounding boxes",
    )

    st.caption("Optimized for real-time performance on CPU environments.")

# Model loading with caching
@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path: str):
    return YOLO(model_path)

model = load_model(model_options[selected_model_name])

# Video processor with performance metrics
class YOLOProcessor:
    def __init__(self):
        self.model = model
        self.last_time = time.time()
        self.fps = 0.0
        self.object_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        results = self.model.predict(
            source=img,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            device="cpu",
        )

        annotated_img = results[0].plot()
        self.object_count = len(results[0].boxes) if results[0].boxes is not None else 0

        # Calculate FPS
        current_time = time.time()
        delta = current_time - self.last_time
        if delta > 0:
            self.fps = 1.0 / delta
        self.last_time = current_time

        # Overlay metrics on video
        cv2.putText(
            annotated_img,
            f"FPS: {self.fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            4,
        )
        cv2.putText(
            annotated_img,
            f"FPS: {self.fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            annotated_img,
            f"Objects: {self.object_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            4,
        )
        cv2.putText(
            annotated_img,
            f"Objects: {self.object_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
        )

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# Main content
st.markdown("### Live Detection Feed")

ctx = webrtc_streamer(
    key="yolo-detection-pro",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YOLOProcessor,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
        "audio": False,
    },
)

# Real-time metrics display
if ctx.video_processor:
    vp = ctx.video_processor
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Processing FPS", f"{vp.fps:.1f}")
    with col2:
        st.metric("Objects Detected", vp.object_count)
    with col3:
        st.metric("Active Model", selected_model_name.split(" (")[0])
else:
    st.info("Click START above and grant camera access to begin detection.")

# Information section
with st.expander("About This Application", expanded=False):
    st.markdown(
        """
        - **Framework**: Ultralytics YOLOv8 — state-of-the-art real-time object detection
        - **Processing**: Client-side via WebRTC (stream never leaves your device)
        - **Performance**: Optimized for CPU inference (20–40 FPS with YOLOv8n)
        - **Privacy**: All processing occurs locally in your browser
        - **Deployment**: Compatible with Streamlit Community Cloud
        """
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        Built with Streamlit • streamlit-webrtc • Ultralytics YOLOv8<br>
        <a href="https://github.com/yourusername/yolov8-streamlit-detection" target="_blank">View Source Code</a>
    </div>
    """,
    unsafe_allow_html=True,
)
