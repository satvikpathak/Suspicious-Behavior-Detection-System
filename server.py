from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import cv2
import json
import sys
import requests
from datetime import datetime
import pytz

# Add src/ to the Python path to import main.py and other modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import your existing Rakshak AI modules
import main
from annotate import annotate_frame
from behavior import detect_behavior
from face_processing import FaceProcessor
from alert import check_alert_conditions, send_alert

app = Flask(__name__)
CORS(app)

# xAI API configuration
XAI_API_KEY = "your-xai-api-key"  # Replace with your API key
XAI_API_URL = "https://api.x.ai/v1/generate"

# Camera data (aligned with frontend structure)
CAMERAS = [
    {
        "id": "CAM_001",
        "name": "Sector 17 Plaza - Main Entrance",
        "location": "Sector 17",
        "country": "India",
        "countryCode": "IN",
        "lat": 30.7398,
        "lng": 76.7827,
        "status": "online",
        "streamUrl": "http://localhost:5000/static/sector17.mp4",
        "thumbnailUrl": None,  # Will be set dynamically
        "manufacturer": "Custom",
        "rating": 4,
        "hasVideo": True,
        "lastSeen": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
        "timezone": "Asia/Kolkata",
        "zip": "160017",
        "city": "Chandigarh",
        "region": "Punjab",
        "source": "custom"
    },
    {
        "id": "CAM_002",
        "name": "Rose Garden - Central Path",
        "location": "Rose Garden",
        "country": "India",
        "countryCode": "IN",
        "lat": 30.7473,
        "lng": 76.7693,
        "status": "online",
        "streamUrl": "http://localhost:5000/static/rose_garden.mp4",
        "thumbnailUrl": None,  # Will be set dynamically
        "manufacturer": "Custom",
        "rating": 3,
        "hasVideo": True,
        "lastSeen": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
        "timezone": "Asia/Kolkata",
        "zip": "160019",
        "city": "Chandigarh",
        "region": "Punjab",
        "source": "custom"
    },
    {
        "id": "CAM_003",
        "name": "Sukhna Lake - Boat Club",
        "location": "Sukhna Lake",
        "country": "India",
        "countryCode": "IN",
        "lat": 30.7421,
        "lng": 76.8188,
        "status": "online",
        "streamUrl": "http://localhost:5000/static/sukhna_lake.mp4",
        "thumbnailUrl": None,  # Will be set dynamically
        "manufacturer": "Custom",
        "rating": 5,
        "hasVideo": True,
        "lastSeen": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
        "timezone": "Asia/Kolkata",
        "zip": "160101",
        "city": "Chandigarh",
        "region": "Punjab",
        "source": "custom"
    },
    {
        "id": "CAM_004",
        "name": "Sector 22 - Market Area",
        "location": "Sector 22",
        "country": "India",
        "countryCode": "IN",
        "lat": 30.7333,
        "lng": 76.7794,
        "status": "maintenance",
        "streamUrl": None,
        "thumbnailUrl": None,
        "manufacturer": "Custom",
        "rating": 2,
        "hasVideo": False,
        "lastSeen": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
        "timezone": "Asia/Kolkata",
        "zip": "160022",
        "city": "Chandigarh",
        "region": "Punjab",
        "source": "custom"
    }
]

# In-memory store for incidents (replace with a database for production)
INCIDENTS = {
    "CAM_001": [],
    "CAM_002": [],
    "CAM_003": [],
    "CAM_004": [],
}

# Initialize FaceProcessor (using HOG model to avoid MemoryError)
face_processor = FaceProcessor()

# Generate thumbnails for each camera's video
def generate_thumbnail(video_path, camera_id):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None
        thumbnail_path = f"static/thumbnails/thumbnail_{camera_id}.jpg"
        os.makedirs("static/thumbnails", exist_ok=True)
        cv2.imwrite(thumbnail_path, frame)
        cap.release()
        return f"http://localhost:5000/static/thumbnails/thumbnail_{camera_id}.jpg"
    except Exception as e:
        print(f"Error generating thumbnail for {camera_id}: {e}")
        return None

# Set thumbnail URLs for all cameras at startup
for camera in CAMERAS:
    if camera["streamUrl"]:
        video_path = os.path.join("static", camera["streamUrl"].split("/")[-1])
        camera["thumbnailUrl"] = generate_thumbnail(video_path, camera["id"])

def process_frame(frame, camera_id):
    camera = next((cam for cam in CAMERAS if cam["id"] == camera_id), None)
    if not camera:
        raise ValueError(f"Camera {camera_id} not found")

    results = main.detect_objects(frame)
    annotated_frame = annotate_frame(frame, results)

    frame_id = 1
    face_data = {}
    tracks = {}
    person_positions = {}
    start_time = datetime.now().timestamp()
    fps = 30

    behavior_alerts, tracks, person_positions = detect_behavior(
        results, frame_id, face_data, tracks, person_positions, start_time, fps
    )

    face_results = face_processor.process_faces(frame, results)
    if face_results:
        face_data.update(face_results)

    alerts = check_alert_conditions(results, face_results)

    for alert in alerts + behavior_alerts:
        send_alert(
            alert,
            frame_id,
            frame,
            face_data,
            camera_name=camera["name"],
            location=camera["location"],
            all_incidents=INCIDENTS[camera_id]
        )

    incidents = []
    current_time = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y%m%d %H%M%S IST")
    for alert in alerts + behavior_alerts:
        incident = {
            "time": current_time.split(" ")[1].split(" ")[0],
            "type": "weapon" if "weapon" in alert.lower() else "suspicious",
            "description": alert
        }
        incidents.append(incident)
        INCIDENTS[camera_id].append(incident)

    return annotated_frame, incidents

@app.route("/api/cameras", methods=["GET"])
def get_cameras():
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
        local_path = os.path.join("static", camera["streamUrl"].split("/")[-1])
        cap = cv2.VideoCapture(local_path)
        if not cap.isOpened():
            return jsonify({"error": "Failed to open video file"}), 500

        ret, frame = cap.read()
        if not ret:
            cap.release()
            return jsonify({"error": "Failed to read frame"}), 500

        annotated_frame, new_incidents = process_frame(frame, camera_id)

        temp_output = f"demo/temp_frame_{camera_id}.jpg"
        os.makedirs("demo", exist_ok=True)
        cv2.imwrite(temp_output, annotated_frame)

        cap.release()

        return jsonify({
            "streamUrl": camera["streamUrl"],
            "thumbnailUrl": camera["thumbnailUrl"],
            "incidents": INCIDENTS[camera_id],
            "frameUrl": f"/static/{temp_output}",
            "aiOverlays": [
                {"label": "WEAPON DETECTED", "color": "red-500"},
                {"label": "EMOTION: ANGRY", "color": "yellow-500"}
            ] if new_incidents else []
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predictive-alerts", methods=["GET"])
def get_predictive_alerts():
    try:
        all_incidents = []
        for camera_id, incidents in INCIDENTS.items():
            camera = next((cam for cam in CAMERAS if cam["id"] == camera_id), None)
            if camera:
                for inc in incidents:
                    all_incidents.append({
                        "camera": camera["name"],
                        "location": camera["location"],
                        "time": inc["time"],
                        "description": inc["description"]
                    })

        incident_texts = [
            f"{inc['camera']} ({inc['location']}) at {inc['time']}: {inc['description']}"
            for inc in all_incidents
        ]
        prompt = (
            "You are an AI assistant for a police surveillance system. Based on the following historical incidents "
            "from CCTV cameras in Chandigarh, predict potential future risks and recommend preventive actions. "
            "Focus on patterns (e.g., repeated incidents in specific locations, types of behaviors). "
            "Provide a concise prediction (2-3 sentences) and a recommended action for each high-risk area.\n\n"
            f"Incidents:\n{'; '.join(incident_texts) if incident_texts else 'No incidents recorded.'}\n\n"
            "Output format:\n- Location: <location>\nPrediction: <prediction>\nAction: <action>\n"
        )

        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-3",
            "prompt": prompt,
            "max_tokens": 300
        }

        response = requests.post(XAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        prediction_text = result.get("text", "Error generating prediction.")

        predictions = []
        lines = prediction_text.split('\n')
        current_prediction = {}
        for line in lines:
            if line.startswith("- Location:"):
                if current_prediction:
                    predictions.append(current_prediction)
                current_prediction = {"location": line.replace("- Location:", "").strip()}
            elif line.startswith("Prediction:"):
                current_prediction["prediction"] = line.replace("Prediction:", "").strip()
            elif line.startswith("Action:"):
                current_prediction["action"] = line.replace("Action:", "").strip()
        if current_prediction:
            predictions.append(current_prediction)

        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")

        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-3",
            "prompt": prompt,
            "max_tokens": 200
        }

        response = requests.post(XAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        reply = result.get("text", "Error generating reply.")

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload-footage", methods=["POST"])
def upload_footage():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    upload_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(upload_path)

    try:
        cap = cv2.VideoCapture(upload_path)
        incidents = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            _, new_incidents = process_frame(frame, camera_id="UPLOAD")
            incidents.extend(new_incidents)
        cap.release()

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

@app.route("/static/<path:filename>")
def serve_static(filename):
    return app.send_static_file(filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)