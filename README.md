# Real-Time Object Detection with YOLOv8 and Streamlit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://ultralytics.com/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A fast, interactive real-time object detection web app using **YOLOv8** (Ultralytics) and **streamlit-webrtc** for live webcam streaming. Prototype in Kaggle/Colab, deploy seamlessly to Streamlit Community Cloud or Hugging Face Spaces.

## Features
- Real-time inference on browser webcam feed
- Adjustable confidence and IoU thresholds
- Optional object tracking with persistent IDs (ByteTrack)
- Lightweight: Uses `yolov8n.pt` (nano model) for smooth CPU performance
- 20-40+ FPS on modest hardware

## Live Demo
Replace with your deployed URL once live:
[Streamlit Link](https://real-time-object-detection-using-yolov8-opencv-lasyyvmcdwxk63q.streamlit.app/))

## Screenshots
<!-- Add GIF or images here for visual appeal -->
![Demo](demos/demo.gif)  <!-- Upload a short webcam demo GIF to /demos -->

## Quick Start (Local)
1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/real-time-object-detection-using-yolov8-opencv.git
   cd real-time-object-detection-using-yolov8-opencv
