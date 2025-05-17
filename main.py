import cv2
import pandas as pd
from ultralytics import YOLO
from scipy.spatial.distance import cdist

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Parameters
video_path = "pexels-timo-volz-5544073.mp4"
distance_threshold = 75
min_crowd_size = 3

# Video setup
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"📽️ Total frames in the video: {total_frames}")

results = []
frame_num = 0

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

    # Log the frame every 10 frames
    if frame_num % 10 == 0:
        results.append({
            "Frame Number": frame_num,
            "Person Count in Crowd": len(unique_person_indices)
        })

# Save final results
df = pd.DataFrame(results)
df.to_csv("crowd___frame.csv", index=False)
print("✅ Saved: crowd_every_10th_frame.csv")
