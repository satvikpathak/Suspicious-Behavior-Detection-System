import cv2
from ultralytics import YOLO
from annotate import annotate_frame
from behavior import detect_behavior
from alert import send_alert
from face_processing import process_faces
import time
import os

def main():
    try:
        model = YOLO('yolov8s.pt')
        cap = cv2.VideoCapture('test_video.mp4')  # Test video
        if not cap.isOpened():
            print("Error: Could not open video")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        os.makedirs('demo', exist_ok=True)
        out = cv2.VideoWriter('demo/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        tracks = {}
        person_positions = {}
        frame_id = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break

            results = model(frame, verbose=False)
            face_data = process_faces(frame, results)  # {track_id: {'name': str, 'emotion': str}}
            alerts, tracks, person_positions = detect_behavior(
                results, frame_id, face_data, tracks, person_positions, start_time, fps
            )
            annotated_frame = annotate_frame(frame, results, alerts, face_data)

            for alert in alerts:
                send_alert(alert, frame_id, frame, face_data)

            cv2.imshow('Surveillance', annotated_frame)
            out.write(annotated_frame)

            frame_id += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()