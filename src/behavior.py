import numpy as np
from collections import defaultdict

def detect_behavior(results, frame_id, face_data, tracks, person_positions, start_time, fps=30):
    alerts = []
    detections = []

    print(f"Frame {frame_id}: Processing {len(face_data)} face(s)")
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for box, score, cls in zip(boxes, scores, classes):
            if score > 0.25:
                class_name = result.names[int(cls)]
                if class_name in ['person', 'backpack', 'suitcase', 'handbag']:
                    detections.append({
                        'class': class_name if class_name != 'handbag' else 'weapon',
                        'box': box,
                        'score': score,
                        'id': len(detections)
                    })
    print(f"Frame {frame_id}: Detected {len(detections)} objects: {[d['class'] for d in detections]}")

    new_tracks = defaultdict(list)
    new_person_positions = defaultdict(list)

    for det in detections:
        centroid = [(det['box'][0] + det['box'][2]) / 2, (det['box'][1] + det['box'][3]) / 2]
        det_class = det['class']
        det_id = det['id']

        matched = False
        for track_id, track_data in tracks.items():
            last_centroid = track_data[-1]['centroid']
            distance = np.sqrt((centroid[0] - last_centroid[0])**2 + (centroid[1] - last_centroid[1])**2)
            if distance < 25:
                new_tracks[track_id].append({
                    'centroid': centroid,
                    'box': det['box'],
                    'class': det_class,
                    'frame_id': frame_id
                })
                if det_class == 'person':
                    new_person_positions[track_id].append(centroid)
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

    for track_id, track_data in new_tracks.items():
        track_class = track_data[-1]['class']
        duration = (frame_id - min([t['frame_id'] for t in track_data])) / fps

        if track_class == 'person' and duration > 5:
            centroids = np.array([t['centroid'] for t in track_data[-int(5*fps):]])
            if len(centroids) > 1 and np.all(np.max(np.abs(centroids - centroids[0]), axis=0) < 15):
                alerts.append(f"Loitering detected (ID: {track_id})")

        if track_class in ['backpack', 'suitcase'] and duration > 3:
            bag_centroid = track_data[-1]['centroid']
            person_nearby = False
            for pid, pos in new_person_positions.items():
                if pos and np.sqrt((bag_centroid[0] - pos[-1][0])**2 + (bag_centroid[1] - pos[-1][1])**2) < 80:
                    person_nearby = True
                    break
            if not person_nearby:
                alerts.append(f"Unattended baggage detected (ID: {track_id})")

        if track_class == 'weapon':
            alerts.append(f"Weapon detected (ID: {track_id})")

        if track_class == 'person':
            box = track_data[-1]['box']
            aspect_ratio = (box[2] - box[0]) / (box[3] - box[1]) if (box[3] - box[1]) > 0 else 1
            if aspect_ratio > 1.5:
                alerts.append(f"Attacking body language detected (ID: {track_id})")
            if len(track_data) > 5:
                centroids = np.array([t['centroid'] for t in track_data[-5:]])
                speed = np.mean(np.sqrt(np.sum(np.diff(centroids, axis=0)**2, axis=1))) * fps
                if speed > 50:
                    for other_id, pos in new_person_positions.items():
                        if other_id != track_id and pos and np.sqrt((centroid[0] - pos[-1][0])**2 + (centroid[1] - pos[-1][1])**2) < 100:
                            alerts.append(f"Attacking approach detected (ID: {track_id})")

        if track_class in ['backpack', 'suitcase'] and len(track_data) > 5:
            centroids = np.array([t['centroid'] for t in track_data[-5:]])
            speed = np.mean(np.sqrt(np.sum(np.diff(centroids, axis=0)**2, axis=1))) * fps
            if speed > 50 and not any(np.sqrt((centroid[0] - pos[-1][0])**2 + (centroid[1] - pos[-1][1])**2) < 80 for pos in new_person_positions.values() if pos):
                alerts.append(f"Theft detected (ID: {track_id})")

    if len(new_person_positions) >= 2:
        speeds = [np.mean(np.sqrt(np.sum(np.diff(np.array(pos[-5:]), axis=0)**2, axis=1))) * fps for pos in new_person_positions.values() if len(pos) > 5]
        if len(speeds) >= 2 and np.mean(speeds) > 30:
            alerts.append("Sudden dispersal detected")

    for track_id, data in face_data.items():
        if data['name'] != "Unknown":
            alerts.append(f"Suspect detected: {data['name']}")
        if data['emotion'] in ['angry', 'fear']:
            alerts.append(f"Suspicious emotion detected: {data['emotion']} (ID: {track_id})")

    return alerts, new_tracks, new_person_positions