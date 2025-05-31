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
        # Initialize YOLO model and FaceProcessor
        model = YOLO('yolov8s.pt')
        face_processor = FaceProcessor()

        # Prompt user to select input source
        print("Select input source:")
        print("1. Webcam (default)")
        print("2. MP4 video file")
        choice = input("Enter your choice (1 or 2): ").strip()

        # Set up video capture based on user choice
        if choice == "2":
            video_path = input("Enter the path to your MP4 video (e.g., videos/sample.mp4): ").strip()
            if not os.path.exists(video_path):
                print(f"Error: Video file '{video_path}' not found")
                return
            cap = cv2.VideoCapture(video_path)
        else:
            video_path = "webcam"
            cap = cv2.VideoCapture(0)  # Default to webcam

        if not cap.isOpened():
            print(f"Error: Could not open {video_path}")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_delay = int(1000 / fps)  # Delay in milliseconds to match FPS

        # Set up output video writer
        os.makedirs('demo', exist_ok=True)
        output_filename = f"demo/output_{os.path.basename(video_path).replace('.mp4', '')}_{int(time.time())}.mp4"
        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        tracks = {}
        person_positions = {}
        frame_id = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print(f"main: End of video or error reading frame at frame {frame_id}")
                break

            # Log frame dimensions for debugging
            print(f"main: Processing frame {frame_id}, dimensions: {frame.shape}")

            results = model(frame, verbose=False)
            if not results:
                print(f"main: YOLO model returned no results at frame {frame_id}")
                frame_id += 1
                continue

            # Process face data every 5th frame
            if frame_id % 5 == 0:
                face_data = face_processor.process_faces(frame, results)
            else:
                face_data = face_processor.face_data

            if face_data is None:
                print(f"main: FaceProcessor returned None at frame {frame_id}")
                frame_id += 1
                continue

            alerts, tracks, person_positions = detect_behavior(
                results, frame_id, face_data, tracks, person_positions, start_time, fps
            )

            annotated_frame = annotate_frame(frame, results, alerts, face_data)

            if annotated_frame is None or annotated_frame.size == 0:
                print(f"main: Warning: Invalid annotated frame at frame {frame_id}, dimensions: {frame.shape if frame is not None else 'None'}")
                frame_id += 1
                continue

            for alert in alerts:
                send_alert(alert, frame_id, frame, face_data)

            cv2.imshow('Surveillance', annotated_frame)
            out.write(annotated_frame)

            frame_id += 1

            # Add delay to simulate real-time playback for both webcam and MP4
            key = cv2.waitKey(frame_delay)
            if key & 0xFF == ord('q'):
                print("Exiting on user command.")
                break

    except Exception as e:
        print(f"main: Error: {e}")
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()