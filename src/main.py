import cv2
from ultralytics import YOLO
from annotate import annotate_frame
from behavior import detect_behavior
from alert import send_alert
import time

def main():
    try:
        # Initialize YOLOv8 model
        model = YOLO('yolov8s.pt')
        
        # Initialize video capture (webcam or video file)
        cap = cv2.VideoCapture(0)  # Change to 'test_video.mp4' for video file
        if not cap.isOpened():
            print("Error: Could not open video source")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if FPS not available

        # Initialize video writer for demo
        out = cv2.VideoWriter('demo/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # Initialize tracking variables
        tracks = {}
        person_positions = {}
        frame_id = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame")
                break

            # Run YOLOv8 inference
            results = model(frame, verbose=False)

            # Annotate frame
            annotated_frame = annotate_frame(frame, results)

            # Detect behaviors
            alerts, tracks, person_positions = detect_behavior(results, frame_id, tracks, start_time, person_positions, fps)

            # Send alerts
            for alert in alerts:
                send_alert(alert, frame_id)

            # Display and save frame
            cv2.imshow('Surveillance', annotated_frame)
            out.write(annotated_frame)

            frame_id += 1

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()