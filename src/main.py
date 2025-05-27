import cv2
from ultralytics import YOLO
from annotate import annotate_frame
from behavior import detect_behavior
from alert import send_alert

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)  # Replace with video file or RTSP URL
tracks = {}
person_positions = {}
frame_id = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = annotate_frame(frame, model)
    alerts, tracks, person_positions = detect_behavior(results, frame_id, tracks, start_time, person_positions)
    for alert in alerts:
        send_alert(alert, frame_id)
    cv2.imshow('Surveillance', annotated_frame)
    frame_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()