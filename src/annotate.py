import cv2
import numpy as np

# Persistent list of person track_ids marked as "Angry"
angry_persons = set()

def annotate_frame(frame, results, alerts, face_data):
    # Check for invalid input frame
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        print("annotate_frame: Invalid input frame")
        return None

    # Check for invalid results or face_data
    if not results or face_data is None:
        print("annotate_frame: Invalid results or face_data")
        return None

    annotated_frame = frame.copy()

    try:
        # Store person track_ids for emotion association
        person_track_ids = {}

        for result in results:
            # Extract boxes, classes, scores, and track_ids safely
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
            classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []
            scores = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
            track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []

            if len(boxes) != len(classes) or len(boxes) != len(scores):
                print("annotate_frame: Mismatch in YOLO results dimensions")
                return None

            for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                if score < 0.5:
                    continue
                class_name = result.names[int(cls)] if result.names else "Unknown"
                track_id = int(track_ids[i]) if i < len(track_ids) else f"temp_{i}"
                x1, y1, x2, y2 = map(int, box)

                if class_name == "person":
                    color = (0, 255, 0)  # Green for people
                    person_track_ids[track_id] = (x1, y1, x2, y2)  # Store person position
                elif class_name in ["handbag", "backpack"]:
                    if class_name == "handbag":
                        color = (0, 0, 255)  # Red for potential weapon (drawn dynamically)
                    else:
                        color = (0, 165, 255)  # Orange for regular baggage
                else:
                    continue

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # Update the set of angry persons based on face_data
        person_emotions = {}  # Track emotions for current frame's persons
        for track_id, data in face_data.items():
            face_center = ((data["bbox"][0] + data["bbox"][2]) / 2, (data["bbox"][1] + data["bbox"][3]) / 2)
            for person_track_id, (px1, py1, px2, py2) in person_track_ids.items():
                if px1 <= face_center[0] <= px2 and py1 <= face_center[1] <= py2:
                    person_emotions[person_track_id] = data["emotion"]
                    if data.get("emotion") == "Angry":
                        angry_persons.add(person_track_id)
                    break

        y_offset = 30
        # Display persistent alerts (e.g., "Potential weapon detected")
        for alert in alerts:
            if "Suspect" in alert:
                color = (255, 0, 255)  # Magenta for suspects
            else:
                color = (0, 255, 255)  # Yellow for other alerts
            cv2.putText(annotated_frame, f"ALERT: {alert}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30

        # Display suspect names from face_data if face is detected
        for track_id, data in face_data.items():
            if data["name"] != "Unknown":
                cv2.putText(annotated_frame, f"Suspect: {data['name']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                y_offset += 30

        # Display emotions for all detected persons
        for person_track_id in person_track_ids:
            # If person is in angry_persons, display "Angry" (persistent)
            if person_track_id in angry_persons:
                cv2.putText(annotated_frame, f"Emotion: Angry (Person {person_track_id})", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                y_offset += 30
            # Otherwise, display their current emotion (e.g., "Neutral") if available
            elif person_track_id in person_emotions:
                emotion = person_emotions[person_track_id]
                cv2.putText(annotated_frame, f"Emotion: {emotion} (Person {person_track_id})", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                y_offset += 30

        # Display persistent "Emotion: Angry" for all persons marked as Angry, even if not currently detected
        for angry_person_id in angry_persons:
            if angry_person_id not in person_track_ids:  # Person not in current frame
                cv2.putText(annotated_frame, f"Emotion: Angry (Person {angry_person_id})", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                y_offset += 30

        return annotated_frame

    except Exception as e:
        print(f"annotate_frame: Error during annotation - {str(e)}")
        return None