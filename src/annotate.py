import cv2
from ultralytics import YOLO

def annotate_frame(frame, model):
    results = model(frame)
    annotated_frame = results[0].plot()  # YOLOv8 plots bounding boxes and labels
    return annotated_frame

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)  # Replace with video file or RTSP URL if available
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    annotated_frame = annotate_frame(frame, model)
    cv2.imshow('Surveillance', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()