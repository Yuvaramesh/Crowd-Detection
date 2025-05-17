import streamlit as st
import cv2
import pandas as pd
import tempfile
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from PIL import Image
import numpy as np
import warnings

# Load YOLO model
model = YOLO("yolov8n.pt")
warnings.filterwarnings("ignore", message=".*use_column_width.*")
# Streamlit App
st.title("👥 Crowd Detection using YOLOv8")
st.markdown("Detects crowded areas in videos based on person proximity.")

# Parameters
distance_threshold = st.slider("Distance Threshold", min_value=30, max_value=150, value=75)
min_crowd_size = st.slider("Minimum Crowd Size", min_value=2, max_value=10, value=3)

uploaded_file = st.file_uploader("📤 Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to temp location
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.info(f"📽️ Total frames in the video: {total_frames}")
    
    results = []
    frame_num = 0

    # Show progress
    progress_bar = st.progress(0)
    frame_display_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Detect persons
        results_yolo = model(frame, classes=[0], verbose=False)[0]
        person_boxes = results_yolo.boxes.xyxy.cpu().numpy() if results_yolo.boxes else []
        centers = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in person_boxes]

        # Group detection
        groups = []
        if len(centers) >= min_crowd_size:
            dists = cdist(centers, centers)
            visited = set()
            for i in range(len(centers)):
                if i not in visited:
                    neighbors = [j for j in range(len(centers)) if dists[i][j] < distance_threshold and i != j]
                    if len(neighbors) >= min_crowd_size - 1:
                        group = {i}
                        queue = neighbors.copy()
                        while queue:
                            current = queue.pop(0)
                            if current not in group:
                                group.add(current)
                                new_neighbors = [j for j in range(len(centers)) if dists[current][j] < distance_threshold and j not in group]
                                queue.extend(new_neighbors)
                        if len(group) >= min_crowd_size:
                            groups.append(group)
                            visited.update(group)

        # Collect unique person indices in all valid groups
        unique_person_indices = set()
        for group in groups:
            unique_person_indices.update(group)

        # Draw crowd boxes and display
        if frame_num % 10 == 0:
            for i, (x1, y1, x2, y2) in enumerate(person_boxes):
                color = (0, 255, 0) if i in unique_person_indices else (255, 0, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            results.append({
                "Frame Number": frame_num,
                "Person Count in Crowd": len(unique_person_indices)
            })

            # Show frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_display_placeholder.image(Image.fromarray(frame_rgb), caption=f"Frame {frame_num}", use_container_width=True)

        # Update progress
        progress = int((frame_num / total_frames) * 100)
        progress_bar.progress(progress)

    cap.release()

    # Save results
    df = pd.DataFrame(results)
    st.success("✅ Processing complete!")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name="crowd_every_10th_frame.csv", mime="text/csv")
