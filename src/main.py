import cv2
import numpy as np
from ultralytics import YOLO
import os
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Load YOLOv9 model (use a lighter model for speed)
object_model = YOLO("yolov9c.pt")  # Switch to yolov9t.pt if available for faster inference

# Load pre-trained ONNX model for emotion detection
try:
    emotion_net = cv2.dnn.readNetFromONNX("emotion_model.onnx")
    print("Successfully loaded emotion_model.onnx")
except cv2.error as e:
    print(f"Failed to load emotion_model.onnx: {e}")
    print("Please ensure the file exists in the src/ directory and is a valid ONNX model.")
    exit()

# Update emotion labels to match emotion-ferplus-8.onnx
EMOTION_LABELS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

def preprocess_frame(frame, target_size=(640, 640)):
    """
    Preprocess frame by resizing and converting to RGB.
    Args:
        frame: Input frame.
        target_size: Tuple of (width, height) for resizing.
    Returns:
        Preprocessed frame.
    """
    resized_frame = cv2.resize(frame, target_size)
    return resized_frame

def detect_objects(frame):
    """
    Detect objects (guns, knives, phones, bags, etc.) in the frame using YOLOv9.
    Args:
        frame: A numpy array (image) from cv2.
    Returns:
        List of detections with bounding boxes, labels, and confidence scores.
    """
    try:
        print(f"Detecting objects in frame of shape: {frame.shape}")
        results = object_model(frame, imgsz=640)  # Use smaller image size for speed
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                label = object_model.names[int(box.cls)]
                if label in ["person", "gun", "knife", "cell phone", "backpack", "bag", "suitcase"]:
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "label": label,
                        "confidence": conf
                    })
        print(f"Detected {len(detections)} objects: {detections}")
        return detections
    except Exception as e:
        print(f"Error in detect_objects: {e}")
        return []

def detect_emotions(frame, person_detections):
    """
    Detect emotions for detected persons using a pre-trained ONNX model.
    Args:
        frame: A numpy array (image) from cv2.
        person_detections: List of person detections.
    Returns:
        List of emotions for detected persons.
    """
    emotions = []
    for person in person_detections:
        x1, y1, x2, y2 = person["bbox"]
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            emotions.append({"bbox": [x1, y1, x2, y2], "emotion": "unknown"})
            continue

        # Preprocess face for emotion detection
        face_img = cv2.resize(face_img, (64, 64))  # Adjusted to match emotion-ferplus-8.onnx input
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
        emotion_net.setInput(blob)
        emotion_probs = emotion_net.forward()
        emotion_idx = np.argmax(emotion_probs[0])
        emotion_label = EMOTION_LABELS[emotion_idx]
        emotions.append({"bbox": [x1, y1, x2, y2], "emotion": emotion_label})

    print(f"Detected emotions: {emotions}")
    return emotions

def detect_threats(detections):
    """
    Detect potential threats based on object detections.
    Args:
        detections: List of object detections.
    Returns:
        List of threat alerts.
    """
    threats = []
    person_detections = [d for d in detections if d["label"] == "person"]
    threat_objects = [d for d in detections if d["label"] in ["gun", "knife", "backpack", "bag", "suitcase"]]

    for person in person_detections:
        px1, py1, px2, py2 = person["bbox"]
        person_center = ((px1 + px2) // 2, (py1 + py2) // 2)

        for obj in threat_objects:
            ox1, oy1, ox2, oy2 = obj["bbox"]
            obj_center = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)

            # Calculate distance between person and object
            dist = np.sqrt((person_center[0] - obj_center[0])**2 + (person_center[1] - obj_center[1])**2)
            
            # If the person is close to a threatening object, flag it as a threat
            if dist < 100:  # Adjust threshold based on your video resolution
                threats.append(f"Potential Threat: Person with {obj['label']}")

    print(f"Threats detected: {threats}")
    return threats

def annotate_frame(frame, detections, threats, emotions):
    """
    Annotate the frame with detections, threats, and emotions.
    Args:
        frame: A numpy array (image) from cv2.
        detections: List of object detections.
        threats: List of threat alerts.
        emotions: List of facial emotions.
    Returns:
        Annotated frame.
    """
    annotated_frame = frame.copy()

    # Draw bounding boxes and labels for detections
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = detection["label"]
        conf = detection["confidence"]
        color = (0, 255, 0) if label == "person" else (0, 0, 255)  # Green for person, red for objects
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated_frame,
            f"{label}: {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # Draw threat alerts near the person
    person_detections = [d for d in detections if d["label"] == "person"]
    for i, person in enumerate(person_detections):
        x1, y1, x2, y2 = person["bbox"]
        for threat in threats:
            if "Person" in threat:  # Assuming threat is tied to a person
                cv2.putText(
                    annotated_frame,
                    threat,
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),  # Red for threats
                    2
                )

    # Draw emotions near the person's face
    for emotion in emotions:
        x1, y1, x2, y2 = emotion["bbox"]
        emotion_label = emotion["emotion"]
        cv2.putText(
            annotated_frame,
            f"Emotion: {emotion_label}",
            (x1, y2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),  # Yellow for emotions
            2
        )

    return annotated_frame

def process_frame(frame):
    """
    Process a frame for objects, threats, and emotions.
    Args:
        frame: A numpy array (image) from cv2.
    Returns:
        Tuple of (detections, threats, emotions).
    """
    # Preprocess frame for faster inference
    processed_frame = preprocess_frame(frame, target_size=(640, 640))
    
    # Detect objects
    detections = detect_objects(processed_frame)
    
    # Detect threats
    threats = detect_threats(detections)
    
    # Detect emotions for persons
    person_detections = [d for d in detections if d["label"] == "person"]
    emotions = detect_emotions(processed_frame, person_detections)
    
    return detections, threats, emotions

# Add a main block for testing with video file and real-time visualization
if __name__ == "__main__":
    # Path to the video file
    video_path = "../videos/sector17.mp4"  # Relative to src/
    
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        exit()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        exit()

    # Process frames from the video and display in a window
    frame_count = 0
    skip_frames = 2  # Process every 2nd frame for speed
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break

        frame_count += 1
        if frame_count % skip_frames != 0:  # Skip frames to speed up
            continue

        print(f"\nProcessing frame {frame_count}")
        
        # Process the frame
        detections, threats, emotions = process_frame(frame)
        
        # Annotate the frame with detections, threats, and emotions
        annotated_frame = annotate_frame(frame, detections, threats, emotions)
        
        # Display the annotated frame in a window
        cv2.imshow("Rakshak AI - Real-Time Threat Detection", annotated_frame)
        
        # Print results for this frame
        print(f"Frame {frame_count} Results - Detections: {detections}")
        print(f"Frame {frame_count} Threats: {threats}")
        print(f"Frame {frame_count} Emotions: {emotions}")

        # Add a small delay and check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing completed")