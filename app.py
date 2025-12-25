import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2

st.set_page_config(
    page_title="YOLOv8 Real-Time Object Detection",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main > div {padding-top: 2rem; padding-bottom: 2rem;}
    .stSlider > div > div > div > div {background-color: #f0f2f6;}
    h1 {font-weight: 600; color: #1e3a8a;}
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    # 🔍 Real-Time Object Detection  
    ### Powered by **YOLOv8** (Ultralytics)
    State-of-the-art detection in your browser webcam. (Tracking disabled for cloud compatibility—detection runs smoothly at 20-40 FPS.)
    """)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # yolov8s.pt for better accuracy if needed

model = load_model()

with st.expander("ℹ️ Model & Info", expanded=False):
    st.markdown("""
        - **Model**: YOLOv8n (nano) – Optimized for CPU real-time.
        - **Mode**: Detection (no tracking IDs for Streamlit Cloud stability).
        - **Tip**: Upgrade to yolov8s/m.pt for higher accuracy (slower on free tier).
        """)

col1, col2 = st.columns([1, 1])
with col1:
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05,
                               help="Lower = more detections (possible false positives).")
with col2:
    iou_threshold = st.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.45, 0.05,
                              help="Higher = fewer overlapping boxes.")

class YOLOProcessor:
    def __init__(self):
        self.model = model
        self.conf = conf_threshold
        self.iou = iou_threshold

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Pure detection (no track/lap needed)
        results = self.model.predict(
            img,
            conf=self.conf,
            iou=self.iou
        )
        
        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

st.markdown("### 📹 Live Webcam Feed")
st.markdown("> Allow camera access. Real-time detection starts instantly.")

webrtc_streamer(
    key="yolo-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YOLOProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.markdown("""
    ---
    Built with ❤️ using **Streamlit**, **streamlit-webrtc**, and **Ultralytics YOLOv8**.  
    [GitHub Repo](https://github.com/itstharusha/Real-time-object-detection-Using-YOLOv8-OpenCV)
    """)
