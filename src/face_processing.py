import face_recognition
import cv2
import numpy as np
import json
import os

class FaceProcessor:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.face_data = {}
        self.person_emotions = {}  # Persistent mapping of person track_id to emotion
        self.load_known_faces()

    def load_known_faces(self):
        if os.path.exists("data/known_suspects.json"):
            with open("data/known_suspects.json", "r") as f:
                data = json.load(f)
                for name, encoding in data.items():
                    self.known_encodings.append(np.array(encoding))
                    self.known_names.append(name)

    def process_faces(self, frame, results):
        self.face_data = {}
        if frame is None or not isinstance(frame, np.ndarray):
            return None

        # Convert frame to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Get YOLO results for person and handbag detection
        person_boxes = []
        handbag_boxes = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []

            for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                if score < 0.5:
                    continue
                class_name = result.names[int(cls)]
                track_id = int(track_ids[i]) if i < len(track_ids) else f"temp_{i}"

                if class_name == "person":
                    person_boxes.append((box, track_id))
                elif class_name == "handbag":
                    handbag_boxes.append(box)

        # Process each detected face
        for idx, (face_encoding, (top, right, bottom, left)) in enumerate(zip(face_encodings, face_locations)):
            track_id = f"face_{idx}"
            name = "Unknown"
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.6)
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_names[first_match_index]

            # Check if this face belongs to a person holding a handbag
            face_center = ((left + right) / 2, (top + bottom) / 2)
            is_holding_weapon = False
            associated_person_id = None

            for person_box, person_track_id in person_boxes:
                px1, py1, px2, py2 = person_box
                person_center = ((px1 + px2) / 2, (py1 + py2) / 2)

                # Check if face is within the person's bounding box
                if (px1 <= face_center[0] <= px2 and py1 <= face_center[1] <= py2):
                    associated_person_id = person_track_id
                    # Check if this person is near a handbag
                    for handbag_box in handbag_boxes:
                        hx1, hy1, hx2, hy2 = handbag_box
                        handbag_center = ((hx1 + hx2) / 2, (hy1 + hy2) / 2)
                        dist_to_handbag = np.linalg.norm(np.array(person_center) - np.array(handbag_center))

                        if dist_to_handbag < 100:  # Proximity threshold
                            is_holding_weapon = True
                            break
                    break

            # Update persistent emotion for the person
            if associated_person_id:
                if is_holding_weapon:
                    self.person_emotions[associated_person_id] = "Angry"  # Set to Angry if holding a weapon
                # If person already marked as Angry, keep it; otherwise, default to Neutral
                emotion = self.person_emotions.get(associated_person_id, "Neutral")

                self.face_data[track_id] = {
                    "name": name,
                    "emotion": emotion,
                    "bbox": [left, top, right, bottom]
                }

        return self.face_data