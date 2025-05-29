import cv2

def annotate_frame(frame, results, alerts, face_data):
    if frame is None or frame.size == 0:
        return None

    annotated_frame = frame.copy()
    height = frame.shape[0]

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            if score > 0.3:
                x1, y1, x2, y2 = map(int, box)
                class_name = result.names[int(cls)]
                color = (0, 255, 0)
                label = f"{class_name} {score:.2f}"

                if class_name in ['backpack', 'suitcase']:
                    color = (0, 165, 255)  # Orange-ish
                elif class_name == 'handbag':
                    color = (0, 0, 255)  # Red
                    label = f"Weapon {score:.2f}"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for i, alert in enumerate(alerts):
        color = (255, 255, 0)  # Default Yellow
        if "Weapon" in alert:
            color = (0, 0, 255)  # Red
        elif "Attacking" in alert:
            color = (255, 0, 0)  # Blue
        elif "Suspect" in alert or "emotion" in alert:
            color = (255, 0, 255)  # Magenta

        cv2.putText(annotated_frame, alert, (10, 30 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for i, (track_id, data) in enumerate(face_data.items()):
        text = f"ID {track_id}: {data['name']}, {data['emotion']}"
        color = (255, 0, 255) if data['name'] != "Unknown" or data['emotion'] in ['angry', 'fear'] else (255, 255, 255)
        cv2.putText(annotated_frame, text, (10, height - 30 - i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return annotated_frame
