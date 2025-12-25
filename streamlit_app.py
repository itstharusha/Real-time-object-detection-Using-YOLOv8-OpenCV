import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from ultralytics import YOLO
import av
import threading

# Page config
st.set_page_config(
    page_title="YOLOv8 Object Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Scoped CSS
st.markdown("""
<style>
    .card {
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .status {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-size: 0.9em;
        font-weight: 600;
    }
    .status.active { background: #ecfdf5; color: #065f46; }
    .status.ready { background: #fef3c7; color: #92400e; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("YOLOv8 Object Detection Dashboard")
st.caption("Real-time object detection powered by Ultralytics YOLOv8")

# Model loading
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
st.success("✅ Model YOLOv8n loaded successfully")

# Layout
left, right = st.columns([1, 2], gap="large")

with left:
    st.container()
    st.subheader("Detection Parameters")
    conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

    with st.expander("Model Information"):
        st.write("""
        - **Architecture**: YOLOv8n (nano)
        - **Inference Mode**: Client-side via WebRTC
        - **Performance**: ~20–35 FPS
        - **Tip**: Use yolov8m/l for higher accuracy
        """)

with right:
    st.container()
    st.subheader("Live Detection Feed")
    st.caption("Click 'Start Camera' and grant permission.")

    class YOLODetector(VideoProcessorBase):
        def __init__(self):
            self.lock = threading.Lock()
            self.conf = conf
            self.iou = iou

        def update_params(self, conf, iou):
            with self.lock:
                self.conf, self.iou = conf, iou

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            with self.lock:
                c, i = self.conf, self.iou
            results = model.predict(img, conf=c, iou=i, verbose=False, device="cpu")
            annotated = results[0].plot(line_width=2, font_size=1)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    ctx = webrtc_streamer(
        key="yolo-feed",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLODetector,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False},
        async_processing=True
    )

    if ctx.video_processor:
        ctx.video_processor.update_params(conf, iou)

    if ctx.state.playing:
        st.markdown('<span class="status active">Live detection active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status ready">Ready – Start Camera</span>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Powered by Streamlit • Ultralytics YOLOv8")
