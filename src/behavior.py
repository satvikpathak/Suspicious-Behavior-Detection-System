import numpy as np
from collections import defaultdict, deque
import time

def detect_behavior(results, frame_id, tracks, start_time, person_positions, fps=30):
    """
    Detect suspicious behaviors: loitering, unattended baggage, sudden dispersal.
    Args:
        results: YOLOv8 detection results
        frame_id: Current frame number
        tracks: Dict tracking object positions over time
        start_time: Start time of processing
        person_positions: Dict of person centroids
        fps: Frames per second (default 30)
    Returns:
        alerts: List of detected behavior alerts
        tracks: Updated tracks
        person_positions: Updated person positions
    """
    alerts = []
    detections = []

    # Extract detections from YOLOv8 results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class IDs
        for box, score, cls in zip(boxes, scores, classes):
            if score > 0.5:  # Confidence threshold
                class_name = result.names[int(cls)]
                if class_name in ['person', 'backpack', 'suitcase', 'handbag']:
                    detections.append({
                        'class': class_name,
                        'box': box,
                        'score': score,
                        'id': len(detections)  # Temporary ID for this frame
                    })

    # Update tracks
    new_tracks = defaultdict(list)
    new_person_positions = defaultdict(deque)

    for det in detections:
        centroid = [(det['box'][0] + det['box'][2]) / 2, (det['box'][1] + det['box'][3]) / 2]
        det_class = det['class']
        det_id = det['id']

        # Match with existing tracks
        matched = False
        for track_id, track_data in tracks.items():
            last_centroid = track_data[-1]['centroid']
            distance = np.sqrt((centroid[0] - last_centroid[0])**2 + (centroid[1] - last_centroid[1])**2)
            if distance < 50:  # Threshold for same object
                new_tracks[track_id].append({
                    'centroid': centroid,
                    'box': det['box'],
                    'class': det_class,
                    'frame_id': frame_id
                })
                if det_class == 'person':
                    new_person_positions[track_id].append(centroid)
                    if len(new_person_positions[track_id]) > 100:
                        new_person_positions[track_id].popleft()
                matched = True
                break

        if not matched:
            new_track_id = f"{det_class}_{frame_id}_{det_id}"
            new_tracks[new_track_id].append({
                'centroid': centroid,
                'box': det['box'],
                'class': det_class,
                'frame_id': frame_id
            })
            if det_class == 'person':
                new_person_positions[new_track_id].append(centroid)

    # Detect behaviors
    for track_id, track_data in new_tracks.items():
        track_class = track_data[-1]['class']
        track_frames = [t['frame_id'] for t in track_data]
        duration = (frame_id - min(track_frames)) / fps  # Duration in seconds

        # Loitering: Person stationary for >15 seconds
        if track_class == 'person' and duration > 15:
            centroids = np.array([t['centroid'] for t in track_data[-int(15*fps):]])
            if len(centroids) > 1:
                movement = np.max(np.abs(centroids - centroids[0]), axis=0)
                if np.all(movement < 20):  # Minimal movement
                    alerts.append(f"Loitering detected (ID: {track_id})")

        # Unattended Baggage: Bag without nearby person for >10 seconds
        if track_class in ['backpack', 'suitcase', 'handbag'] and duration > 10:
            bag_centroid = track_data[-1]['centroid']
            person_nearby = False
            for pid, pos in new_person_positions.items():
                if pos:
                    person_centroid = pos[-1]
                    distance = np.sqrt((bag_centroid[0] - person_centroid[0])**2 + (bag_centroid[1] - person_centroid[1])**2)
                    if distance < 100:  # Person within 100 pixels
                        person_nearby = True
                        break
            if not person_nearby:
                alerts.append(f"Unattended baggage detected (ID: {track_id})")

    # Sudden Dispersal: Multiple people moving away rapidly
    if len(new_person_positions) >= 3:  # Need at least 3 people
        recent_positions = {pid: pos for pid, pos in new_person_positions.items() if len(pos) > 10}
        if len(recent_positions) >= 3:
            speeds = []
            for pid, pos in recent_positions.items():
                pos_array = np.array(list(pos)[-10:])  # Last 10 frames (~0.33s)
                if len(pos_array) > 1:
                    deltas = np.diff(pos_array, axis=0)
                    speed = np.mean(np.sqrt(np.sum(deltas**2, axis=1))) * fps  # Pixels/second
                    speeds.append(speed)
            if len(speeds) >= 3 and np.mean(speeds) > 50:  # High average speed
                alerts.append("Sudden dispersal detected")

    return alerts, new_tracks, new_person_positions