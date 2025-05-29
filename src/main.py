import cv2
from ultralytics import YOLO
from annotate import annotate_frame
from behavior import detect_behavior
from alert import send_alert
from face_processing import FaceProcessor
import time
import os

def main():
    cap = None
    out = None
    try:
        model = YOLO('yolov8s.pt')
        face_processor = FaceProcessor()

        cap = cv2.VideoCapture(0)  # Use webcam (device 0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
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
            if not ret or frame is None or frame.size == 0:
                print("Error: Could not read valid frame from webcam")
                continue

            results = model(frame, verbose=False)

            # Process face data every 5th frame
            if frame_id % 5 == 0:
                face_data = face_processor.process_faces(frame, results)
            else:
                face_data = face_processor.face_data

            alerts, tracks, person_positions = detect_behavior(
                results, frame_id, face_data, tracks, person_positions, start_time, fps
            )

            annotated_frame = annotate_frame(frame, results, alerts, face_data)

            # ✅ Add check before imshow/write
            if annotated_frame is None or annotated_frame.size == 0:
                print(f"Warning: Invalid annotated frame at frame {frame_id}")
                frame_id += 1
                continue


            for alert in alerts:
                send_alert(alert, frame_id, frame, face_data)

            cv2.imshow('Surveillance', annotated_frame)
            out.write(annotated_frame)

            frame_id += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting on user command.")
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
