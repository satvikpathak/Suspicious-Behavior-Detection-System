import face_recognition
import cv2
import json
import os
import numpy as np

class FaceProcessor:
    def __init__(self):
        self.face_data = {}
        json_path = 'data/known_suspects.json'
        os.makedirs('data', exist_ok=True)
        if not os.path.exists(json_path):
            with open(json_path, 'w') as f:
                json.dump({}, f)
        with open(json_path, 'r') as f:
            self.known_face_encodings = json.load(f)

    def process_faces(self, frame, results):
        self.face_data = {}
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        person_boxes = []
        track_ids = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for box, cls, score in zip(boxes, classes, scores):
                if score > 0.3 and result.names[int(cls)] == 'person':
                    person_boxes.append(box)
                    track_ids.append(f"person_{len(person_boxes)}")
        for box, track_id in zip(person_boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            face_locations = face_recognition.face_locations(rgb_frame[y1:y2, x1:x2])
            face_encodings = face_recognition.face_encodings(rgb_frame[y1:y2, x1:x2], face_locations)
            name = "Unknown"
            for face_encoding in face_encodings:
                for known_name, known_encoding in self.known_face_encodings.items():
                    match = face_recognition.compare_faces([np.array(known_encoding)], face_encoding)
                    if match[0]:
                        name = known_name
                        break
            self.face_data[track_id] = {"name": name, "emotion": "Neutral"}
        return self.face_data