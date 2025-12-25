%%writefile app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2

st.title("Real-Time Object Detection with YOLOv8")

# Load model (cache for performance)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Change to custom model if trained

model = load_model()

# Sidebar controls
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45)

class YOLOProcessor:
    def __init__(self):
        self.model = model
        self.conf = conf_threshold
        self.iou = iou_threshold

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # YOLO inference (use track for persistence across frames)
        results = self.model.track(
            img,
            conf=self.conf,
            iou=self.iou,
            persist=True,
            tracker="bytetrack.yaml"  # Optional: for object tracking IDs
        )
        
        annotated = results[0].plot()  # Draw boxes/labels
        
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# WebRTC streamer
webrtc_streamer(
    key="yolo-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YOLOProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True  # Crucial for smooth real-time performance
)
