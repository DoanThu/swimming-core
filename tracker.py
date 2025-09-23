from ultralytics import YOLO

# Load an official or custom model
model = YOLO("models/pose-1080-2025.pt")  # Load an official Pose model

# Perform tracking with the model
results = model.track("DJI_0304_resized.MP4", show=True, verbose=False, save=True, imgsz=640)  # Tracking with default tracker
# results = model.track("https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # with ByteTrack