import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from ultralytics import YOLO
import av
import threading
import pandas as pd

# Import ByteTrack from YOLOX (you need to install yolox)
from yolox.tracker.byte_tracker import BYTETracker
from ultralytics.utils.plotting import Annotator

# Page config
st.set_page_config(
    page_title="YOLOv8 Object Tracking Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Scoped CSS
st.markdown("""
<style>
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
st.title("YOLOv8 Object Tracking Dashboard")
st.caption("Real-time detection + tracking powered by Ultralytics YOLOv8 + ByteTrack")

# Model loading
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
st.success("✅ Model YOLOv8n loaded successfully")

# Layout
left, right = st.columns([1, 2], gap="large")

with left:
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
    st.subheader("Live Tracking Feed")
    st.caption("Click 'Start Camera' and grant permission.")

    class YOLODetector(VideoProcessorBase):
        def __init__(self):
            self.lock = threading.Lock()
            self.conf = conf
            self.iou = iou
            self.tracker = BYTETracker()   # persistent tracker
            self.last_tracks = []

        def update_params(self, conf, iou):
            with self.lock:
                self.conf, self.iou = conf, iou

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            with self.lock:
                c, i = self.conf, self.iou

            # Run YOLO detection
            results = model.predict(img, conf=c, iou=i, verbose=False, device="cpu")
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                dets = boxes.xywh.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy()

                # Update tracker
                tracks = self.tracker.update(dets, scores, cls_ids, img.shape)

                # Draw boxes with IDs
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
        st.markdown('<span class="status active">Live tracking active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status ready">Ready – Start Camera</span>', unsafe_allow_html=True)

# Sidebar: tracked objects table
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
st.caption("Powered by Streamlit • Ultralytics YOLOv8 + ByteTrack")
