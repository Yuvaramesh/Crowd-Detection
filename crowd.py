import cv2
import pandas as pd
from ultralytics import YOLO
from scipy.spatial.distance import cdist
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Lightweight model; change as needed

# Parameters
video_path = "dataset_video.mp4"
distance_threshold = 75  # in pixels
min_crowd_size = 3
min_crowd_duration = 10  # in consecutive frames

# Initialize
cap = cv2.VideoCapture(video_path)
frame_number = 0
crowd_candidates = {}  # key: group ID, value: list of (frame_number, person_centers)

results_list = []
group_id_counter = 0

def are_groups_similar(group1, group2, threshold=distance_threshold):
    """Check if two groups (lists of centers) are spatially similar."""
    if len(group1) != len(group2):
        return False
    dists = cdist(group1, group2)
    match_count = sum((dists.min(axis=1) < threshold))
    return match_count >= len(group1) * 0.7  # 70% match

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # Detect persons
    results = model(frame, classes=[0], verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
    centers = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in boxes]

    # Skip if fewer than min crowd size
    if len(centers) < min_crowd_size:
        crowd_candidates.clear()
        continue

    dists = cdist(centers, centers)
    added_to_group = set()

    # Identify new crowd groups in this frame
    for i, center in enumerate(centers):
        if i in added_to_group:
            continue
        neighbors = [j for j in range(len(centers)) if dists[i][j] < distance_threshold and i != j]
        if len(neighbors) >= min_crowd_size - 1:
            group = [centers[j] for j in [i] + neighbors]
            matched = False

            # Check if group matches any existing group
            for gid in list(crowd_candidates):
                last_frame, last_group = crowd_candidates[gid][-1]
                if frame_number - last_frame <= 1 and are_groups_similar(group, last_group):
                    crowd_candidates[gid].append((frame_number, group))
                    matched = True
                    break

            if not matched:
                # Create new group ID
                group_id_counter += 1
                crowd_candidates[group_id_counter] = [(frame_number, group)]

            for j in neighbors:
                added_to_group.add(j)
            added_to_group.add(i)

    # Check and finalize persistent groups
    for gid in list(crowd_candidates):
        frames_data = crowd_candidates[gid]
        if frame_number - frames_data[-1][0] > 1:
            if len(frames_data) >= min_crowd_duration:
                start_frame = frames_data[0][0]
                avg_count = int(np.mean([len(g) for _, g in frames_data]))
                results_list.append({
                    "Frame Number": start_frame,
                    "Person Count in Crowd": avg_count
                })
            del crowd_candidates[gid]

# Final check for lingering crowds at end
for gid in list(crowd_candidates):
    frames_data = crowd_candidates[gid]
    if len(frames_data) >= min_crowd_duration:
        start_frame = frames_data[0][0]
        avg_count = int(np.mean([len(g) for _, g in frames_data]))
        results_list.append({
            "Frame Number": start_frame,
            "Person Count in Crowd": avg_count
        })

# Save results
df = pd.DataFrame(results_list)
df.to_csv("crowd_detection_results.csv", index=False)

cap.release()
print("✅ Crowd detection completed. Results saved to 'crowd_detection_results.csv'.")
