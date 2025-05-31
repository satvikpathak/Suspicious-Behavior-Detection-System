import numpy as np
import cv2
from collections import defaultdict, deque

# Global flag to track if a weapon has ever been detected
weapon_detected = False

def detect_behavior(results, frame_id, face_data, tracks, person_positions, start_time, fps):
    global weapon_detected
    alerts = []
    current_time = frame_id / fps

    # Store person bounding boxes for proximity checks
    person_boxes = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []

        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            if score < 0.5:  # General confidence threshold
                continue
            class_name = result.names[int(cls)]
            track_id = int(track_ids[i]) if i < len(track_ids) else f"temp_{frame_id}_{i}"

            if class_name == "person":
                x1, y1, x2, y2 = map(int, box)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                if track_id not in person_positions:
                    person_positions[track_id] = deque(maxlen=30)
                person_positions[track_id].append(center)

                if len(person_positions[track_id]) > 15:
                    movement = np.std(list(person_positions[track_id]), axis=0)
                    if movement[0] < 10 and movement[1] < 10 and current_time > 3:
                        alerts.append(f"Loitering detected for person {track_id}")

                # Store person box for proximity check
                person_boxes.append((x1, y1, x2, y2, track_id))

            elif class_name in ["handbag", "backpack"]:
                tracks[track_id] = tracks.get(track_id, {"last_seen": frame_id, "bbox": box, "near_person": False})
                tracks[track_id]["last_seen"] = frame_id
                tracks[track_id]["bbox"] = box

                # Hardcode: Treat every 'handbag' as a potential weapon
                if class_name == "handbag":
                    weapon_detected = True  # Set flag once a weapon is detected

                # Check for unattended baggage
                near_person = False
                for person_box in person_boxes:
                    px1, py1, px2, py2, _ = person_box
                    dist = np.linalg.norm(np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]) -
                                         np.array([(px1 + px2) / 2, (py1 + py2) / 2]))
                    if dist < 100:
                        near_person = True
                        break

                tracks[track_id]["near_person"] = near_person
                if not near_person and frame_id - tracks[track_id].get("first_unattended", frame_id) > fps * 2:
                    tracks[track_id]["first_unattended"] = tracks[track_id].get("first_unattended", frame_id)
                    alerts.append(f"Unattended baggage detected at {track_id}")

    # Persist the weapon alert if a weapon was ever detected
    if weapon_detected:
        alerts.append("Potential weapon detected")

    for track_id, data in list(tracks.items()):
        if frame_id - data["last_seen"] > fps * 5:
            del tracks[track_id]

    person_counts = defaultdict(int)
    for box, cls, score in zip(boxes, classes, scores):
        if score < 0.5:
            continue
        if result.names[int(cls)] == "person":
            person_counts[frame_id] += 1

    if frame_id > fps and person_counts[frame_id] < person_counts[frame_id - fps] / 2:
        alerts.append("Sudden dispersal detected")

    for track_id, data in face_data.items():
        if data["name"] != "Unknown":
            alerts.append(f"Suspect {data['name']} detected")

    return alerts, tracks, person_positions