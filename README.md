# Suspicious Behavior Detection System
Detects loitering, unattended baggage, sudden dispersal, attacking behavior, theft, and suspicious faces using webcam or MP4 video input.
## Demo
See `demo/output_*.mp4` and `demo/frame_*.jpg`.
## Setup
```bash
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
# Place MP4 videos in videos/ folder (optional)
python src/main.py