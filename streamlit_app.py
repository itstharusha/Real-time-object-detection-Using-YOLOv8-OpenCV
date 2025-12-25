import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2
import threading
import time

# Page Configuration
st.set_page_config(
    page_title="YOLOv8 Object Tracking Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Custom CSS for Professional Dashboard Aesthetic
st.markdown("""
<style>
    /* Professional system font stack */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }

    /* Container */
    .block-container {
        max-width: 1400px;
        padding-top: 3rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Hide Streamlit elements */
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
        font-weight: 400;
    }

    /* Cards */
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

    /* Video container */
    .video-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        background: #000;
        margin-top: 1rem;
    }

    /* Status indicators */
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

    /* Sidebar */
    .sidebar-table {
        font-size: 0.875rem;
    }

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

    /* Responsive */
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
    <h1>YOLOv8 Object Tracking Dashboard</h1>
    <p>Real-time detection and persistent object tracking using Ultralytics YOLOv8 with ByteTrack</p>
</div>
""", unsafe_allow_html=True)

# Model Loading
@st.cache_resource(show_spinner=False)
def load_model():
    with st.spinner("Initializing YOLOv8 model..."):
        return YOLO("yolov8n.pt")

model = load_model()

st.markdown("""
<div class="status-indicator">
    Model Status: YOLOv8n (nano) loaded successfully
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Shared state for tracked objects and FPS
if "tracked_objects" not in st.session_state:
    st.session_state.tracked_objects = []
if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()

# Layout
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Tracking Parameters</div>", unsafe_allow_html=True)

    conf = st.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.25, 0.05,
        help="Minimum confidence for a detection to be considered"
    )

    iou = st.slider(
        "IoU Threshold (NMS)",
        0.0, 1.0, 0.45, 0.05,
        help="IoU threshold for non-maximum suppression"
    )

    track_thresh = st.slider(
        "Track Threshold",
        0.0, 1.0, 0.5, 0.1,
        help="Higher values make tracking more strict (default: 0.5)"
    )

    match_thresh = st.slider(
        "Match Threshold",
        0.0, 1.0, 0.8, 0.1,
        help="Matching threshold for associating detections (default: 0.8)"
    )

    with st.expander("Model & Tracker Information"):
        st.markdown("""
        - **Model**: YOLOv8n (nano) – optimized for real-time inference
        - **Tracker**: ByteTrack (built-in Ultralytics)
        - **Inference**: Client-side via WebRTC
        - **Expected Performance**: 20–35 FPS on modern hardware
        - **Tip**: Use yolov8m.pt or yolov8l.pt for higher accuracy
        """)

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Live Tracking Feed</div>", unsafe_allow_html=True)

    st.caption("Select 'Start Camera' and allow access. All parameters update in real time.")

    class YOLOTracker(VideoProcessorBase):
        def __init__(self):
            self.lock = threading.Lock()
            self.conf = conf
            self.iou = iou
            self.track_thresh = track_thresh
            self.match_thresh = match_thresh
            self.frame_count = 0
            self.start_time = time.time()

        def update_params(self, conf, iou, track_thresh, match_thresh):
            with self.lock:
                self.conf = conf
                self.iou = iou
                self.track_thresh = track_thresh
                self.match_thresh = match_thresh

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            with self.lock:
                c_conf = self.conf
                c_iou = self.iou
                c_track_thresh = self.track_thresh
                c_match_thresh = self.match_thresh

            # Run tracking with custom thresholds
            results = model.track(
                source=img,
                conf=c_conf,
                iou=c_iou,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                track_high_thresh=c_track_thresh,
                track_low_thresh=0.1,
                match_thresh=c_match_thresh,
                device="cpu"
            )[0]

            # Manual annotation for reliable visibility
            annotated_img = img.copy()

            tracked_objects = []
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results.boxes.id.cpu().numpy().astype(int)
                classes = results.boxes.cls.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()

                for box, track_id, cls_id, conf_score in zip(boxes, track_ids, classes, confs):
                    x1, y1, x2, y2 = box
                    label = f"{model.names[cls_id]} ID:{track_id} {conf_score:.2f}"

                    # Draw box and label
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )

                    tracked_objects.append({
                        "ID": int(track_id),
                        "Class": model.names[cls_id],
                        "Confidence": f"{conf_score:.2f}"
                    })

            # Update session state (thread-safe)
            st.session_state.tracked_objects = tracked_objects
            st.session_state.last_update = time.time()

            # FPS calculation
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                annotated_img, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
            )

            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

    ctx = webrtc_streamer(
        key="yolov8-tracking",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOTracker,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
            "audio": False
        },
        async_processing=True,
        translations={"start": "Start Camera", "stop": "Stop Camera"}
    )

    if ctx.video_processor:
        ctx.video_processor.update_params(conf, iou, track_thresh, match_thresh)

    if ctx.state.playing:
        st.markdown("<div class='status-indicator status-active'>Live tracking active</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-indicator status-ready'>Ready – Select 'Start Camera'</div>", unsafe_allow_html=True)

    st.markdown("<div class='video-container'></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Sidebar: Live Tracked Objects Table
st.sidebar.header("Currently Tracked Objects")

if ctx and ctx.state.playing and st.session_state.tracked_objects:
    df = pd.DataFrame(st.session_state.tracked_objects)
    st.sidebar.dataframe(df.sort_values("ID"), use_container_width=True)

    # Auto-refresh hint
    last_update = st.session_state.last_update
    if time.time() - last_update < 1.0:
        st.sidebar.caption("Updated live")
    else:
        st.sidebar.caption("Waiting for new frame...")
else:
    st.sidebar.info("No active tracks – start camera and point at objects")

# Footer
st.markdown("""
<div class="footer">
    Powered by Streamlit • Ultralytics YOLOv8 with ByteTrack<br>
    <a href="https://ultralytics.com" target="_blank">Ultralytics</a> •
    <a href="https://docs.ultralytics.com/modes/track" target="_blank">Tracking Documentation</a>
</div>
""", unsafe_allow_html=True)
