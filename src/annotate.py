import cv2
import numpy as np

def annotate_frame(frame, results, alerts, face_data):
    if frame is None or frame.size == 0:
        return frame

    annotated_frame = frame.copy()

    # Store person boxes for proximity check
    person_boxes = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []

        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            if score < 0.5:
                continue
            class_name = result.names[int(cls)]
            track_id = int(track_ids[i]) if i < len(track_ids) else f"temp_{frame_id}_{i}"
            x1, y1, x2, y2 = map(int, box)

            if class_name == "person":
                color = (0, 255, 0)  # Green for people
                person_boxes.append((x1, y1, x2, y2))
            elif class_name in ["handbag", "backpack"]:
                # Check if handbag is in a person's hand
                handbag_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                in_hand = False

                for px1, py1, px2, py2 in person_boxes:
                    hand_area_y = py2 - (py2 - py1) * 0.3  # Bottom 30% of person box
                    hand_area_x1 = px1
                    hand_area_x2 = px2

                    if (hand_area_x1 <= handbag_center[0] <= hand_area_x2 and
                            hand_area_y <= handbag_center[1] <= py2):
                        in_hand = True
                        break

                if class_name == "handbag" and in_hand:
                    color = (0, 0, 255)  # Red for potential weapon in hand
                else:
                    color = (0, 165, 255)  # Orange for regular baggage
            else:
                continue

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

    y_offset = 30
    for alert in alerts:
        if "Suspect" in alert:
            color = (255, 0, 255)  # Magenta for suspects
        else:
            color = (0, 255, 255)  # Yellow for other alerts
        cv2.putText(annotated_frame, f"ALERT: {alert}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30

    for track_id, data in face_data.items():
        if data["name"] != "Unknown":
            cv2.putText(annotated_frame, f"Suspect: {data['name']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            y_offset += 30
        cv2.putText(annotated_frame, f"Emotion: {data['emotion']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        y_offset += 30

    return annotated_frame