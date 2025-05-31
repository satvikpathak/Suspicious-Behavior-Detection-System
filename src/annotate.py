import cv2
import numpy as np

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
                elif class_name in ["handbag", "backpack"]:
                    # Hardcode: Treat every 'handbag' as a potential weapon
                    if class_name == "handbag":
                        color = (0, 0, 255)  # Red for potential weapon
                    else:
                        color = (0, 165, 255)  # Orange for regular baggage (e.g., backpack)
                else:
                    continue

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        y_offset = 30
        # Ensure persistent alerts are displayed (e.g., "Potential weapon detected")
        for alert in alerts:
            if "Suspect" in alert:
                color = (255, 0, 255)  # Magenta for suspects
            else:
                color = (0, 255, 255)  # Yellow for other alerts
            cv2.putText(annotated_frame, f"ALERT: {alert}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30

        # Display persistent emotions for detected faces
        for track_id, data in face_data.items():
            if data["name"] != "Unknown":
                cv2.putText(annotated_frame, f"Suspect: {data['name']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                y_offset += 30
            # Display the persistent emotion ("Angry" or "Neutral")
            cv2.putText(annotated_frame, f"Emotion: {data['emotion']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            y_offset += 30

        return annotated_frame

    except Exception as e:
        print(f"annotate_frame: Error during annotation - {str(e)}")
        return None