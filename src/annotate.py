import cv2
import numpy as np

def annotate_frame(frame, results):
    """
    Annotate frame with bounding boxes and labels from YOLOv8 results.
    Args:
        frame: Input frame (numpy array)
        results: YOLOv8 detection results
    Returns:
        annotated_frame: Frame with annotations
    """
    annotated_frame = frame.copy()

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            if score > 0.5:
                x1, y1, x2, y2 = map(int, box)
                class_name = result.names[int(cls)]
                
                # Default color (green) for normal detections
                color = (0, 255, 0)
                label = f"{class_name} {score:.2f}"

                # Highlight suspicious objects (placeholder for behavior linkage)
                if class_name in ['backpack', 'suitcase', 'handbag']:
                    color = (0, 0, 255)  # Red for potential unattended baggage
                elif class_name == 'person':
                    # Could add logic to link with behavior alerts
                    pass

                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated_frame