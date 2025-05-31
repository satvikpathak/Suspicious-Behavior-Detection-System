# Suspicious Behavior Detection System 🚨

**Team Cypher | Cytherthon.ai Hackathon – May 2025**

A real-time AI surveillance system that detects suspicious behaviors, faces, and potential weapons using webcam or MP4 input. Built from scratch using YOLOv8, OpenCV, and `face_recognition`.

---

## 👥 Team Details

- **Team Name:** Cypher  
- **Team Leader:** Satvik Pathak  
- **Members:** Shivam Vats, Ryanveer Singh, Sanatan Sharma

---

## 🔍 Features

### 🎯 Behavior Detection
- **Loitering:** Flags stationary individuals.
- **Unattended Baggage:** Detects left bags without owners.
- **Sudden Dispersal:** Triggers when crowd size drops quickly.

### 🧨 Weapon Detection (Demo)
- Handbags are treated as proxies for weapons.
- Red bounding box follows handbag.
- Alert persists once triggered.

### 🧠 Face Recognition & Emotion Labeling
- Recognizes known suspects via `data/known_suspects.json`.
- Shows suspect names in **magenta**.
- Labels:  
  - Weapon holder → **"Angry"**  
  - Others → **"Neutral"**

### 🎥 Input & Output
- Webcam or MP4 (user selected).
- Outputs:
  - Processed video → `demo/output_*.mp4`
  - Alert snapshots → `demo/frame_*.jpg`

---

## ⚙️ Setup Instructions

### 📁 Clone & Navigate
```bash
git clone <your-repo-url>
cd suspicious-behavior-detection
```

### 🚀 Create Virtual Environment
python -m venv env
.\env\Scripts\activate  # For Windows
#### source env/bin/activate  # For Linux/Mac

### 📦 Install Dependencies
Create a requirements.txt with:
```bash
ultralytics
opencv-python==4.10.0.84
numpy==1.26.4
torch==2.3.1
torchvision==0.18.1
pillow==10.4.0
face_recognition
reportlab
dlib==19.22.1
typing-extensions
```

pip install -r requirements.txt

### 📂 Prepare Data
Place known suspect images in data/suspect_images/

Generate encodings into data/known_suspects.json

(Optional) Add test videos in videos/

## ▶️ Run the Project
```bash
python src/main.py
```
Select input source when prompted:

1 → Use webcam

2 → Enter path to video (e.g., videos/sample.mp4)

Press q to quit and save results.

## 📂 Project Structure
```bash
suspicious-behavior-detection/
├── data/
│   ├── known_suspects.json
│   └── suspect_images/
├── demo/
│   ├── output_*.mp4
│   └── frame_*.jpg
├── src/
│   ├── main.py
│   ├── annotate.py
│   ├── behavior.py
│   ├── face_processing.py
│   └── alert.py
├── videos/
├── requirements.txt
└── README.md
```
### 🛠 Troubleshooting
No face detected? Check lighting and make sure JSON encoding is correct.

Weapon not detected? Adjust YOLO confidence threshold in behavior.py.

Video not found? Double-check path and file name.

Slow detection? Use lighter YOLOv8 model like yolov8n.pt.

### 🚀 Future Improvements

Improve real-time emotion detection.

Improve object and face tracking.

Support multi-camera inputs and remote feeds.

# Built with ❤️ by Team Cypher