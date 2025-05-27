import numpy as np
from collections import defaultdict
import time

def detect_behavior(results, frame_id, tracks, start_time, person_positions):
    # Extract detections
    detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
    current_time = time.time()
    
    # Initialize tracks for new objects
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        cls_name = results[0].names[int(cls)]
        obj_id = f"{cls_name}_{frame_id}_{centroid[0]}"  # Simple ID based on position
        tracks[obj_id] = tracks.get(obj_id, {'centroid': centroid, 'frames': 0, 'last_seen': current_time})
        tracks[obj_id]['frames'] += 1
        if cls_name == 'person':
            person_positions[obj_id] = centroid
    
    # Check for suspicious behaviors
    alerts = []
    for obj_id, track in tracks.items():
        cls_name = obj_id.split('_')[0]
        duration = current_time - track['last_seen']
        
        # Loitering: Person stationary for >30 seconds
        if cls_name == 'person' and track['frames'] > 30 * 30:  # Assuming 30 fps
            alerts.append(f"Loitering detected for {obj_id}")
        
        # Unattended baggage: Bag with no person nearby for >20 seconds
        if cls_name == 'bag' and duration > 20:
            nearby = False
            bag_centroid = track['centroid']
            for person_id, person_centroid in person_positions.items():
                dist = np.sqrt((bag_centroid[0] - person_centroid[0])**2 + (bag_centroid[1] - person_centroid[1])**2)
                if dist < 100:  # Pixel distance threshold
                    nearby = True
                    break
            if not nearby:
                alerts.append(f"Unattended bag detected for {obj_id}")
    
    # Sudden dispersal: Drop in person count
    current_person_count = sum(1 for obj_id in tracks if obj_id.startswith('person'))
    if frame_id > 30 and person_positions.get('prev_count', 0) > 3 and current_person_count < person_positions.get('prev_count', 0) - 3:
        alerts.append("Sudden dispersal detected")
    person_positions['prev_count'] = current_person_count
    
    return alerts, tracks, person_positions