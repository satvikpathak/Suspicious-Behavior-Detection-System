from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import cv2
import subprocess
import json
import sys
from datetime import datetime
import pytz

# Add src/ to the Python path to import main.py and other modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import your existing Rakshak AI modules
import main
from annotate import annotate_frame
from behavior import detect_behavior
from face_processing import FaceProcessor
from alert import check_alert_conditions, send_alert  # Updated import to include send_alert

app = Flask(__name__)
CORS(app)

# Camera data (replace with real Chandigarh Police data)
CAMERAS = [
    {
        "id": "CAM_001",
        "name": "Sector 17 Plaza - Main Entrance",
        "location": "Sector 17",
        "lat": 30.7398,
        "lng": 76.7827,
        "status": "online",
        "streamUrl": "rtsp://example.com/sector17_main"
    },
    {
        "id": "CAM_002",
        "name": "Rose Garden - Central Path",
        "location": "Rose Garden",
        "lat": 30.7473,
        "lng": 76.7693,
        "status": "online",
        "streamUrl": "rtsp://example.com/rose_garden"
    },
    {
        "id": "CAM_003",
        "name": "Sukhna Lake - Boat Club",
        "location": "Sukhna Lake",
        "lat": 30.7421,
        "lng": 76.8188,
        "status": "online",
        "streamUrl": "rtsp://example.com/sukhna_lake"
    },
    {
        "id": "CAM_004",
        "name": "Sector 22 - Market Area",
        "location": "Sector 22",
        "lat": 30.7333,
        "lng": 76.7794,
        "status": "maintenance",
        "streamUrl": None
    }
]

# In-memory store for incidents (replace with a database for production)
INCIDENTS = {
    "CAM_001": [],
    "CAM_002": [],
    "CAM_003": [],
    "CAM_004": [],
}

# Initialize FaceProcessor
face_processor = FaceProcessor()

def process_frame(frame, camera_id):
    """Process a single frame using Rakshak AI's pipeline."""
    # Detect people and objects (using YOLOv8 from main.py)
    results = main.detect_objects(frame)  # Assumes main.py has a detect_objects function

    # Annotate frame with bounding boxes and labels
    annotated_frame = annotate_frame(frame, results)

    # Detect suspicious behavior
    # Initialize required arguments for detect_behavior
    frame_id = 1  # Placeholder; in a real stream, this would increment per frame
    face_data = {}  # Placeholder; will be populated by process_faces
    tracks = {}
    person_positions = {}
    start_time = datetime.now().timestamp()
    fps = 30  # Assumed FPS; adjust based on your video source

    behavior_alerts, tracks, person_positions = detect_behavior(
        results, frame_id, face_data, tracks, person_positions, start_time, fps
    )

    # Recognize faces using FaceProcessor
    face_results = face_processor.process_faces(frame, results)
    if face_results:
        face_data.update(face_results)

    # Check for alerts (e.g., potential weapon)
    alerts = check_alert_conditions(results, face_results)

    # Generate reports for each alert
    for alert in alerts + behavior_alerts:
        send_alert(alert, frame_id, frame, face_data)

    # Combine results
    incidents = []
    current_time = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y%m%d %H%M%S IST")
    for alert in alerts + behavior_alerts:
        incident = {
            "time": current_time.split(" ")[1].split(" ")[0],  # e.g., "23:43"
            "type": "weapon" if "weapon" in alert.lower() else "suspicious",
            "description": alert
        }
        incidents.append(incident)
        INCIDENTS[camera_id].append(incident)

    return annotated_frame, incidents

@app.route("/api/cameras", methods=["GET"])
def get_cameras():
    # Return the list of cameras with their latest incidents
    cameras_with_incidents = [
        {**camera, "incidents": INCIDENTS[camera["id"]]}
        for camera in CAMERAS
    ]
    return jsonify(cameras_with_incidents)

@app.route("/api/live-feed/<camera_id>", methods=["GET"])
def get_live_feed(camera_id):
    camera = next((cam for cam in CAMERAS if cam["id"] == camera_id), None)
    if not camera:
        return jsonify({"error": "Camera not found"}), 404

    if camera["status"] != "online":
        return jsonify({"error": f"Camera is {camera['status']}"}, 503)

    try:
        # Open RTSP stream
        cap = cv2.VideoCapture(camera["streamUrl"])
        if not cap.isOpened():
            return jsonify({"error": "Failed to open RTSP stream"}), 500

        # Read a single frame for demo purposes
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return jsonify({"error": "Failed to read frame"}), 500

        # Process the frame using Rakshak AI
        annotated_frame, new_incidents = process_frame(frame, camera_id)

        # Save the annotated frame temporarily
        temp_output = f"demo/temp_frame_{camera_id}.jpg"
        os.makedirs("demo", exist_ok=True)
        cv2.imwrite(temp_output, annotated_frame)

        # Release the capture
        cap.release()

        # Return the feed URL and incidents
        return jsonify({
            "streamUrl": camera["streamUrl"],
            "incidents": INCIDENTS[camera_id],
            "frameUrl": f"/static/{temp_output}",
            "aiOverlays": [
                {"label": "WEAPON DETECTED", "color": "red-500"},
                {"label": "EMOTION: ANGRY", "color": "yellow-500"}
            ] if new_incidents else []
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload-footage", methods=["POST"])
def upload_footage():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file temporarily
    upload_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(upload_path)

    try:
        # Process the video using Rakshak AI's main.py
        cap = cv2.VideoCapture(upload_path)
        incidents = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            _, new_incidents = process_frame(frame, camera_id="UPLOAD")
            incidents.extend(new_incidents)
        cap.release()

        # Clean up the uploaded file
        if os.path.exists(upload_path):
            os.remove(upload_path)

        return jsonify({"result": "Analysis complete", "incidents": incidents})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/incidents", methods=["GET"])
def get_incidents():
    all_incidents = []
    for camera_id, incidents in INCIDENTS.items():
        all_incidents.extend(incidents)
    return jsonify(all_incidents)

# Serve static files (e.g., annotated frames)
@app.route("/static/<path:filename>")
def serve_static(filename):
    return app.send_static_file(filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)