import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from ultralytics import YOLO
import av
import threading

# Page Configuration
st.set_page_config(
    page_title="YOLOv8 Object Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# Custom CSS for Professional Dashboard Aesthetic
st.markdown("""
<style>
    /* System font stack for maximum professionalism and performance */
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }

    /* Container adjustments */
    .block-container {
        max-width: 1400px;
        padding-top: 3rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Header */
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 600;
        color: #111827;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.125rem;
        color: #4b5563;
        text-align: center;
        margin-top: 0;
        font-weight: 400;
    }

    /* Clean card containers */
    .dashboard-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        height: 100%;
    }

    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }

    /* Video feed container */
    .video-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        background: #000;
    }

    /* Status indicators - subtle and professional */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.375rem 0.75rem;
        border-radius: 6px;
        font-size: 0.875rem;
        font-weight: 500;
        background: #f3f4f6;
        color: #374151;
    }
    .status-active { background: #ecfdf5; color: #065f46; }
    .status-ready { background: #fef3c7; color: #92400e; }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding: 1.5rem 0;
        color: #6b7280;
        font-size: 0.875rem;
        border-top: 1px solid #e5e7eb;
    }
    .footer a {
        color: #4f46e5;
        text-decoration: none;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        div[data-testid="column"] {
            width: 100% !important;
            margin-bottom: 2rem;
        }
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>YOLOv8 Object Detection Dashboard</h1>
    <p>Real-time object detection powered by Ultralytics YOLOv8</p>
</div>
""", unsafe_allow_html=True)

# Model Loading
@st.cache_resource(show_spinner=False)
def load_model():
    with st.spinner("Initializing model..."):
        return YOLO("yolov8n.pt")

model = load_model()

st.markdown("""
<div class="status-indicator">
    Model Status: YOLOv8n (nano) loaded successfully
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Layout
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Detection Parameters</div>", unsafe_allow_html=True)

    conf = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Filters detections below the specified confidence level"
    )

    iou = st.slider(
        "IoU Threshold (Non-Maximum Suppression)",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Controls suppression of overlapping bounding boxes"
    )

    with st.expander("Model Information"):
        st.markdown("""
        - **Architecture**: YOLOv8n (nano variant) – optimized for real-time performance
        - **Inference Mode**: Client-side via WebRTC
        - **Performance**: 20–35 FPS on standard hardware
        - **Note**: For improved accuracy, consider yolov8m.pt or yolov8l.pt
        """)

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Live Detection Feed</div>", unsafe_allow_html=True)

    st.caption("Select 'Start Camera' and grant permission. Parameter adjustments apply in real time.")

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

            results = model.predict(
                source=img,
                conf=current_conf,
                iou=current_iou,
                verbose=False,
                device="cpu"
            )

            annotated = results[0].plot(
                line_width=2,
                font_size=1,
                labels=True,
                boxes=True,
                probs=True
            )

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    ctx = webrtc_streamer(
        key="yolov8-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLODetector,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
            "audio": False
        },
        async_processing=True,
        translations={
            "start": "Start Camera",
            "stop": "Stop Camera"
        }
    )

    if ctx.video_processor:
        ctx.video_processor.update_params(conf, iou)

    if ctx.state.playing:
        st.markdown("<div class='status-indicator status-active'>Live detection active</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-indicator status-ready'>Ready – Select 'Start Camera'</div>", unsafe_allow_html=True)

    st.markdown("<div class='video-container'>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Powered by Streamlit and Ultralytics YOLOv8<br>
    <a href="https://ultralytics.com" target="_blank">Ultralytics</a> • 
    <a href="https://docs.ultralytics.com" target="_blank">Documentation</a>
</div>
""", unsafe_allow_html=True)
