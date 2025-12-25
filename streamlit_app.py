import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from ultralytics.utils.plotting import Annotator
import av
import threading
import time
import pandas as pd

# Page config
st.set_page_config(
    page_title="VisionGuard AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
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
st.title("VisionGuard AI")
st.subheader("Dashboard Settings")

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Sidebar: Detection Parameters
st.sidebar.header("Detection Parameters")
st.sidebar.write("Fine-tune YOLOv8 model behavior for optimal detection")

conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou = st.sidebar.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.45, 0.05)

st.sidebar.markdown("**Current Configuration**")
st.sidebar.write(f"Confidence Threshold: `{conf}`")
st.sidebar.write(f"IoU Threshold: `{iou}`")

with st.sidebar.expander("Quick Reference Guide"):
    st.write("""
- **Low (0.0–0.3)**: Maximum sensitivity, catches everything
- **Medium (0.3–0.7)**: Balanced approach, recommended
- **High (0.7–1.0)**: Strict filtering, high precision
""")

# Main layout
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Live Detection Stream")
    st.caption("Click 'Start Stream' to begin real-time detection")

    class YOLODetector(VideoProcessorBase):
        def __init__(self):
            self.lock = threading.Lock()
            self.conf = conf
            self.iou = iou
            self.tracker = BYTETracker()
            self.last_tracks = []
            self.last_time = time.time()
            self.fps = 0
            self.latency = 0
            self.start_time = time.time()

        def update_params(self, conf, iou):
            with self.lock:
                self.conf, self.iou = conf, iou

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            start = time.time()
            img = frame.to_ndarray(format="bgr24")

            with self.lock:
                c, i = self.conf, self.iou

            results = model.predict(img, conf=c, iou=i, verbose=False, device="cpu")
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                dets = boxes.xywh.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy()
                tracks = self.tracker.update(dets, scores, cls_ids, img.shape)
                annotator = Annotator(img)
                for track in tracks:
                    x1, y1, x2, y2, track_id, cls_id = track
                    label = f"{model.names[int(cls_id)]} ID:{int(track_id)}"
                    annotator.box_label([x1, y1, x2, y2], label)
                self.last_tracks = tracks
                annotated = annotator.result()
            else:
                self.last_tracks = []
                annotated = img

            # Metrics
            self.latency = int((time.time() - start) * 1000)
            self.fps = int(1 / (time.time() - self.last_time))
            self.last_time = time.time()

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    ctx = webrtc_streamer(
        key="visionguard-stream",
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
        st.markdown('<span class="status ready">Ready – Start Stream</span>', unsafe_allow_html=True)

with right:
    st.subheader("Real-time Metrics")
    if ctx and ctx.video_processor:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Current FPS", ctx.video_processor.fps)
        st.metric("Avg Latency", f"{ctx.video_processor.latency} ms")
        st.metric("Objects Detected", len(ctx.video_processor.last_tracks))
        active_time = int((time.time() - ctx.video_processor.start_time) / 60)
        st.metric("Active Time", f"{active_time} min")
        st.metric("Model", "YOLOv8n")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Stream not active")

    st.subheader("Quick Controls")
    st.write(f"Confidence Threshold: `{conf}`")
    st.write(f"IoU Threshold: `{iou}`")

# Sidebar: Tracked objects
st.sidebar.header("Tracked Objects")
if ctx and ctx.video_processor and ctx.video_processor.last_tracks is not None:
    tracks = ctx.video_processor.last_tracks
    if len(tracks) > 0:
        data = []
        for track in tracks:
            x1, y1, x2, y2, track_id, cls_id = track
            data.append({
                "Track ID": int(track_id),
                "Class": model.names.get(int(cls_id), "unknown"),
            })
        df = pd.DataFrame(data)
        st.sidebar.dataframe(df, use_container_width=True)
    else:
        st.sidebar.write("No objects detected.")
else:
    st.sidebar.write("Camera not active.")

# Footer
st.markdown("---")
st.caption("Powered by YOLOv8 • VisionGuard AI • Streamlit")
