from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Nano model for speed

# Test on a sample video or webcam
cap = cv2.VideoCapture(0)  # Use 0 for webcam or path to video file
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()  # Draw bounding boxes
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()