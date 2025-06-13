import cv2
import numpy as np
from ultralytics import YOLO
import os
import logging
import math
from collections import defaultdict, deque
import onnxruntime as ort
import glob

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Load YOLOv8 model for comprehensive object detection
object_model = YOLO("yolov8n.pt")  # Use yolov8s.pt or yolov8m.pt for better accuracy

# Load YOLOv8 model for pose estimation
pose_model = YOLO("yolov8n-pose.pt")

# Load FER+ model for emotion detection
try:
    emotion_model = ort.InferenceSession("emotion_model.onnx")
    EMOTION_LABELS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
    EMOTION_MODEL_AVAILABLE = True
    print("FER+ emotion model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load FER+ model: {e}")
    EMOTION_MODEL_AVAILABLE = False

# Enhanced weapon categories with specific weapon types
WEAPON_CLASSES = {
    'gun': ['gun', 'pistol', 'rifle', 'firearm', 'handgun', 'revolver', 'shotgun', 'weapon'],
    'knife': ['knife', 'blade', 'sword', 'dagger', 'machete', 'cleaver'],
    'blunt_weapon': ['bat', 'stick', 'club', 'hammer', 'baton', 'pipe'],
    'explosive': ['bomb', 'grenade', 'explosive', 'dynamite']
}

# Enhanced weapon detection keywords for better accuracy
WEAPON_KEYWORDS = {
    'handgun': ['gun', 'pistol', 'handgun', 'revolver', 'firearm'],
    'rifle': ['rifle', 'shotgun', 'carbine', 'assault'],
    'knife': ['knife', 'blade', 'dagger', 'machete'],
    'blunt': ['bat', 'club', 'hammer', 'baton', 'stick']
}

# Suspicious objects that could be used as weapons
SUSPICIOUS_OBJECTS = ['bottle', 'scissors', 'axe', 'chainsaw', 'tool', 'screwdriver', 'wrench']

# Common objects for context (to avoid false positives)
COMMON_OBJECTS = ['cell phone', 'phone', 'laptop', 'bag', 'backpack', 'book', 'cup', 'remote', 'mouse']

# Threat level colors with gradient (1-100 scale)
def get_threat_color(threat_score):
    """Get color based on threat score (1-100)"""
    if threat_score >= 90:
        return (0, 0, 139)    # Dark Red
    elif threat_score >= 80:
        return (0, 0, 255)    # Red
    elif threat_score >= 70:
        return (0, 69, 255)   # Orange Red
    elif threat_score >= 60:
        return (0, 140, 255)  # Dark Orange
    elif threat_score >= 50:
        return (0, 165, 255)  # Orange
    elif threat_score >= 40:
        return (0, 215, 255)  # Gold
    elif threat_score >= 30:
        return (0, 255, 255)  # Yellow
    elif threat_score >= 20:
        return (0, 255, 127)  # Spring Green
    elif threat_score >= 10:
        return (127, 255, 0)  # Chartreuse
    else:
        return (0, 255, 0)    # Green

def get_threat_level_text(threat_score):
    """Get threat level text based on score"""
    if threat_score >= 90:
        return "CRITICAL"
    elif threat_score >= 80:
        return "SEVERE"
    elif threat_score >= 70:
        return "HIGH"
    elif threat_score >= 60:
        return "ELEVATED"
    elif threat_score >= 50:
        return "MODERATE"
    elif threat_score >= 40:
        return "GUARDED"
    elif threat_score >= 30:
        return "CAUTIOUS"
    elif threat_score >= 20:
        return "LOW"
    elif threat_score >= 10:
        return "MINIMAL"
    else:
        return "NORMAL"

class StabilizedTracker:
    """Handles tracking and stabilization of detections"""
    
    def __init__(self, stability_threshold=5, max_age=15):
        self.stability_threshold = stability_threshold
        self.max_age = max_age
        self.tracked_objects = {}
        self.next_id = 0
        
    def update(self, detections):
        """Update tracked objects with new detections"""
        for obj in self.tracked_objects.values():
            obj.updated = False
        
        matched_detections = []
        for detection in detections:
            best_match_id = self._find_best_match(detection)
            
            if best_match_id is not None:
                self.tracked_objects[best_match_id].update(detection)
                matched_detections.append(best_match_id)
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracked_objects[new_id] = TrackedObject(new_id, detection)
                matched_detections.append(new_id)
        
        objects_to_remove = []
        for obj_id, obj in self.tracked_objects.items():
            if not obj.updated:
                obj.age += 1
                if obj.age > self.max_age:
                    objects_to_remove.append(obj_id)
        
        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]
        
        return self.get_stable_detections()
    
    def _find_best_match(self, detection, iou_threshold=0.3):
        """Find the best matching tracked object for a detection"""
        best_match_id = None
        best_iou = 0
        
        det_box = detection["bbox"]
        
        for obj_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj.category == detection["category"]:
                iou = self._calculate_iou(det_box, tracked_obj.last_bbox)
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match_id = obj_id
        
        return best_match_id
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def get_stable_detections(self):
        """Return only stable detections"""
        stable_detections = []
        for obj in self.tracked_objects.values():
            if obj.is_stable(self.stability_threshold):
                stable_detections.append(obj.to_detection())
        return stable_detections

class TrackedObject:
    """Represents a tracked object with stability metrics"""
    
    def __init__(self, obj_id, detection):
        self.id = obj_id
        self.category = detection["category"]
        self.label = detection["label"]
        self.threat_score = detection["threat_score"]
        self.confidence_history = deque([detection["confidence"]], maxlen=10)
        self.bbox_history = deque([detection["bbox"]], maxlen=10)
        self.last_bbox = detection["bbox"]
        self.detection_count = 1
        self.age = 0
        self.updated = True
        
    def update(self, detection):
        """Update tracked object with new detection"""
        self.confidence_history.append(detection["confidence"])
        self.bbox_history.append(detection["bbox"])
        self.last_bbox = detection["bbox"]
        self.detection_count += 1
        self.age = 0
        self.updated = True
        self.threat_score = detection["threat_score"]
    
    def is_stable(self, threshold):
        """Check if object is stable enough to display"""
        return self.detection_count >= threshold
    
    def get_average_confidence(self):
        """Get average confidence from recent detections"""
        return sum(self.confidence_history) / len(self.confidence_history)
    
    def get_stable_bbox(self):
        """Get stabilized bounding box"""
        if len(self.bbox_history) == 1:
            return self.bbox_history[0]
        
        recent_boxes = list(self.bbox_history)[-3:]
        avg_box = [
            int(sum(box[i] for box in recent_boxes) / len(recent_boxes))
            for i in range(4)
        ]
        return avg_box
    
    def to_detection(self):
        """Convert to detection format"""
        return {
            "bbox": self.get_stable_bbox(),
            "label": self.label,
            "category": self.category,
            "confidence": self.get_average_confidence(),
            "threat_score": self.threat_score
        }

class BehaviorStabilizer:
    """Stabilizes behavior and emotion detections"""
    
    def __init__(self, stability_frames=8, cooldown_frames=15):
        self.stability_frames = stability_frames
        self.cooldown_frames = cooldown_frames
        self.behavior_history = defaultdict(int)
        self.active_behaviors = set()
        self.behavior_cooldown = defaultdict(int)
        self.emotion_history = defaultdict(lambda: deque(maxlen=10))
        
    def update_behaviors(self, behaviors):
        """Update and stabilize behavior detections"""
        current_behaviors = set(behaviors)
        
        for behavior in current_behaviors:
            self.behavior_history[behavior] += 1
        
        for behavior in list(self.behavior_history.keys()):
            if behavior not in current_behaviors:
                self.behavior_history[behavior] = max(0, self.behavior_history[behavior] - 1)
        
        new_active = set()
        for behavior, count in self.behavior_history.items():
            if count >= self.stability_frames:
                new_active.add(behavior)
                self.behavior_cooldown[behavior] = self.cooldown_frames
        
        for behavior in list(self.active_behaviors):
            if behavior not in new_active and self.behavior_cooldown[behavior] > 0:
                self.behavior_cooldown[behavior] -= 1
                if self.behavior_cooldown[behavior] > 0:
                    new_active.add(behavior)
        
        self.active_behaviors = new_active
        return list(self.active_behaviors)
    
    def update_emotions(self, emotions):
        """Update and stabilize emotion detections"""
        stable_emotions = []
        
        for emotion_data in emotions:
            person_key = f"{emotion_data['bbox'][0]}_{emotion_data['bbox'][1]}"
            emotion = emotion_data['emotion']
            confidence = emotion_data['confidence']
            
            self.emotion_history[person_key].append((emotion, confidence, emotion_data['threat_emotion']))
            
            if len(self.emotion_history[person_key]) >= 5:
                recent_emotions = list(self.emotion_history[person_key])[-5:]
                
                emotion_counts = defaultdict(int)
                threat_counts = defaultdict(int)
                confidence_sum = defaultdict(float)
                
                for emo, conf, threat in recent_emotions:
                    emotion_counts[emo] += 1
                    confidence_sum[emo] += conf
                    if threat:
                        threat_counts[emo] += 1
                
                most_common_emotion = max(emotion_counts, key=emotion_counts.get)
                avg_confidence = confidence_sum[most_common_emotion] / emotion_counts[most_common_emotion]
                is_threat = threat_counts[most_common_emotion] >= 2
                
                if emotion_counts[most_common_emotion] >= 3:
                    stable_emotions.append({
                        "bbox": emotion_data["bbox"],
                        "emotion": most_common_emotion,
                        "confidence": avg_confidence,
                        "threat_emotion": is_threat
                    })
        
        return stable_emotions

class EnhancedThreatDetector:
    def __init__(self):
        self.frame_history = []
        self.threat_history = []
        self.max_history = 10
        
        self.object_tracker = StabilizedTracker(stability_threshold=3, max_age=10)
        self.behavior_stabilizer = BehaviorStabilizer(stability_frames=5, cooldown_frames=10)
        
    def detect_objects(self, frame):
        """Enhanced object detection with improved weapon classification"""
        try:
            results = object_model(frame, conf=0.2, iou=0.5)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf)
                        class_id = int(box.cls)
                        label = object_model.names[class_id]
                        
                        # Enhanced categorization and threat scoring
                        category, specific_type = self.categorize_object_enhanced(label)
                        threat_score = self.calculate_threat_score(label, category, conf)
                        
                        # Use specific weapon type if identified
                        display_label = specific_type if specific_type else label
                        
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "label": display_label,
                            "category": category,
                            "confidence": conf,
                            "threat_score": threat_score
                        })
            
            return detections
        except Exception as e:
            print(f"Error in detect_objects: {e}")
            return []

    def categorize_object_enhanced(self, label):
        """Enhanced object categorization with specific weapon types"""
        label_lower = label.lower()
        
        # Check for specific weapon types
        for weapon_type, keywords in WEAPON_KEYWORDS.items():
            if any(keyword in label_lower for keyword in keywords):
                return f"weapon_{weapon_type}", weapon_type.upper()
        
        # Check general weapon categories
        for weapon_type, keywords in WEAPON_CLASSES.items():
            if any(keyword in label_lower for keyword in keywords):
                return f"weapon_{weapon_type}", weapon_type.upper()
        
        # Check suspicious objects
        if any(obj in label_lower for obj in SUSPICIOUS_OBJECTS):
            return "suspicious_object", None
        
        # Check common objects to avoid false positives
        if any(obj in label_lower for obj in COMMON_OBJECTS):
            return "common_object", None
        
        if label_lower == "person":
            return "person", None
        
        return "other", None

    def calculate_threat_score(self, label, category, confidence):
        """Calculate threat score (1-100) based on object type and confidence"""
        base_score = 0
        label_lower = label.lower()
        
        # Weapon-specific scoring
        if "weapon_gun" in category or "weapon_handgun" in category:
            base_score = 95  # Firearms are highest threat
        elif "weapon_rifle" in category:
            base_score = 98  # Rifles even higher
        elif "weapon_knife" in category:
            base_score = 75  # Knives are high threat
        elif "weapon_blunt" in category:
            base_score = 60  # Blunt weapons moderate-high
        elif category == "suspicious_object":
            base_score = 35  # Suspicious objects moderate-low
        elif category == "person":
            base_score = 5   # People are minimal threat by default
        else:
            base_score = 1   # Everything else is minimal
        
        # Adjust based on confidence
        confidence_multiplier = min(confidence * 1.2, 1.0)  # Cap at 1.0
        final_score = int(base_score * confidence_multiplier)
        
        return min(max(final_score, 1), 100)  # Ensure score is between 1-100

    def detect_poses(self, frame):
        """Enhanced pose detection"""
        try:
            results = pose_model(frame, conf=0.3)
            poses = []
            
            for result in results:
                if result.keypoints is not None:
                    for person_keypoints in result.keypoints.xy:
                        keypoints = person_keypoints.cpu().numpy()
                        if len(keypoints) >= 17:
                            poses.append(keypoints)
            
            return poses
        except Exception as e:
            print(f"Error in detect_poses: {e}")
            return []

    def analyze_body_language(self, poses, detections):
        """Enhanced body language analysis with threat scoring"""
        behaviors = []
        person_detections = [d for d in detections if d["category"] == "person"]
        
        for i, pose in enumerate(poses):
            if i >= len(person_detections):
                break
                
            person = person_detections[i]
            pose_behaviors = self.analyze_single_pose(pose)
            
            # Check for weapons near person
            weapon_nearby, weapon_type = self.check_weapons_nearby_enhanced(person, detections)
            
            for behavior in pose_behaviors:
                threat_score = self.calculate_behavior_threat_score(behavior, weapon_nearby, weapon_type)
                
                if weapon_nearby:
                    behavior_text = f"ARMED ({weapon_type}): {behavior} [Score: {threat_score}]"
                else:
                    behavior_text = f"{behavior} [Score: {threat_score}]"
                
                behaviors.append(behavior_text)
        
        return behaviors

    def analyze_single_pose(self, keypoints):
        """Analyze individual pose for threatening behaviors"""
        behaviors = []
        
        # Extract key body parts
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        # Validate keypoints
        valid_points = {
            'left_shoulder': left_shoulder[0] > 0,
            'right_shoulder': right_shoulder[0] > 0,
            'left_wrist': left_wrist[0] > 0,
            'right_wrist': right_wrist[0] > 0,
            'left_elbow': left_elbow[0] > 0,
            'right_elbow': right_elbow[0] > 0
        }
        
        # Enhanced pose analysis
        if valid_points['left_shoulder'] and valid_points['left_wrist']:
            if left_wrist[1] < left_shoulder[1] - 40:
                behaviors.append("Raised Left Arm")
        
        if valid_points['right_shoulder'] and valid_points['right_wrist']:
            if right_wrist[1] < right_shoulder[1] - 40:
                behaviors.append("Raised Right Arm")
        
        # Fighting stance
        if all(valid_points[key] for key in ['left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist']):
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            wrists_raised = (left_wrist[1] < shoulder_center_y + 30) and (right_wrist[1] < shoulder_center_y + 30)
            
            if wrists_raised:
                behaviors.append("Fighting Stance")
        
        # Shooting stance
        if all(valid_points[key] for key in ['left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist']):
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            wrist_center_x = (left_wrist[0] + right_wrist[0]) / 2
            
            if abs(wrist_center_x - shoulder_center_x) > 50:
                behaviors.append("Aiming Stance")
        
        # Aggressive posture
        if left_ankle[0] > 0 and right_ankle[0] > 0 and left_hip[0] > 0 and right_hip[0] > 0:
            stance_width = abs(left_ankle[0] - right_ankle[0])
            hip_width = abs(left_hip[0] - right_hip[0])
            
            if stance_width > hip_width * 1.8:
                behaviors.append("Wide Aggressive Stance")
        
        # Lunging motion
        if nose[0] > 0 and left_hip[0] > 0 and right_hip[0] > 0:
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            if abs(nose[0] - hip_center_x) > 40:
                behaviors.append("Lunging Motion")
        
        return behaviors

    def calculate_behavior_threat_score(self, behavior, weapon_nearby, weapon_type):
        """Calculate threat score for behaviors"""
        base_scores = {
            "Raised Left Arm": 20,
            "Raised Right Arm": 20,
            "Fighting Stance": 70,
            "Aiming Stance": 85,
            "Wide Aggressive Stance": 60,
            "Lunging Motion": 75
        }
        
        base_score = base_scores.get(behavior, 10)
        
        # Weapon multipliers
        if weapon_nearby:
            if weapon_type in ["HANDGUN", "RIFLE"]:
                base_score = min(base_score * 1.5, 100)
            elif weapon_type == "KNIFE":
                base_score = min(base_score * 1.3, 100)
            else:
                base_score = min(base_score * 1.2, 100)
        
        return int(base_score)

    def check_weapons_nearby_enhanced(self, person, detections, threshold=150):
        """Enhanced weapon detection near person"""
        px1, py1, px2, py2 = person["bbox"]
        person_center = ((px1 + px2) // 2, (py1 + py2) // 2)
        
        weapon_detections = [d for d in detections if "weapon" in d["category"]]
        
        for weapon in weapon_detections:
            wx1, wy1, wx2, wy2 = weapon["bbox"]
            weapon_center = ((wx1 + wx2) // 2, (wy1 + wy2) // 2)
            distance = math.sqrt((person_center[0] - weapon_center[0])**2 + 
                               (person_center[1] - weapon_center[1])**2)
            
            if distance < threshold:
                weapon_type = weapon["label"].upper()
                return True, weapon_type
        
        return False, None

    def detect_emotions_ferplus(self, frame, person_detections):
        """Detect emotions using FER+ ONNX model"""
        emotions = []
        
        if not EMOTION_MODEL_AVAILABLE:
            return self.detect_emotions_basic(frame, person_detections)
        
        for person in person_detections:
            try:
                x1, y1, x2, y2 = person["bbox"]
                
                # Extract and preprocess face region
                face_height = int((y2 - y1) * 0.4)  # Upper 40% for face
                face_y1 = max(0, y1)
                face_y2 = min(frame.shape[0], y1 + face_height)
                face_x1 = max(0, x1)
                face_x2 = min(frame.shape[1], x2)
                
                face_img = frame[face_y1:face_y2, face_x1:face_x2]
                
                if face_img.size == 0:
                    emotions.append({
                        "bbox": [x1, y1, x2, y2],
                        "emotion": "undetected",
                        "confidence": 0.0,
                        "threat_emotion": False
                    })
                    continue
                
                # Preprocess for FER+
                face_resized = cv2.resize(face_img, (64, 64))
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                face_normalized = face_gray.astype(np.float32) / 255.0
                face_input = np.expand_dims(np.expand_dims(face_normalized, axis=0), axis=0)
                
                # Run inference
                input_name = emotion_model.get_inputs()[0].name
                output = emotion_model.run(None, {input_name: face_input})[0]
                
                # Get emotion probabilities
                emotion_probs = output[0]
                max_idx = np.argmax(emotion_probs)
                emotion = EMOTION_LABELS[max_idx]
                confidence = float(emotion_probs[max_idx] * 100)
                
                # Determine if emotion is threatening
                threat_emotions = ['anger', 'disgust', 'contempt']
                is_threat = emotion in threat_emotions and confidence > 60
                
                emotions.append({
                    "bbox": [x1, y1, x2, y2],
                    "emotion": emotion,
                    "confidence": confidence,
                    "threat_emotion": is_threat
                })
                
            except Exception as e:
                print(f"Error in FER+ emotion detection: {e}")
                emotions.append({
                    "bbox": person["bbox"],
                    "emotion": "error",
                    "confidence": 0.0,
                    "threat_emotion": False
                })
        
        return emotions

    def detect_emotions_basic(self, frame, person_detections):
        """Basic emotion detection fallback"""
        emotions = []
        
        for person in person_detections:
            try:
                x1, y1, x2, y2 = person["bbox"]
                face_height = (y2 - y1) // 3
                face_y1 = max(0, y1)
                face_y2 = min(frame.shape[0], y1 + face_height)
                face_x1 = max(0, x1)
                face_x2 = min(frame.shape[1], x2)
                
                face_img = frame[face_y1:face_y2, face_x1:face_x2]
                
                if face_img.size == 0:
                    continue
                
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray_face)
                contrast = np.std(gray_face)
                
                if contrast > 40 and brightness < 100:
                    emotion = "anger"
                    threat_emotion = True
                    confidence = 65.0
                elif contrast < 20:
                    emotion = "neutral"
                    threat_emotion = False
                    confidence = 70.0
                else:
                    emotion = "neutral"
                    threat_emotion = False
                    confidence = 50.0
                
                emotions.append({
                    "bbox": [x1, y1, x2, y2],
                    "emotion": emotion,
                    "confidence": confidence,
                    "threat_emotion": threat_emotion
                })
                
            except Exception as e:
                print(f"Error in basic emotion detection: {e}")
        
        return emotions

    def annotate_frame(self, frame, detections, behaviors, emotions):
        """Enhanced frame annotation with threat score visualization"""
        annotated_frame = frame.copy()
        
        # Draw object detections with threat score colors
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            label = detection["label"]
            conf = detection["confidence"]
            threat_score = detection["threat_score"]
            
            # Get color based on threat score
            color = get_threat_color(threat_score)
            threat_text = get_threat_level_text(threat_score)
            
            # Draw bounding box with thickness based on threat
            thickness = max(2, min(6, threat_score // 15))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            label_text = f"{label}: {conf:.2f} (T:{threat_score})"
            if threat_score >= 50:
                label_text = f"[{threat_text}] {label_text}"
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw behavior alerts
        alert_y = 30
        for behavior in behaviors:
            # Extract threat score from behavior text
            threat_score = 30  # default
            if "[Score:" in behavior:
                try:
                    score_part = behavior.split("[Score:")[1].split("]")[0]
                    threat_score = int(score_part)
                except:
                    pass
            
            color = get_threat_color(threat_score)
            
            # Draw behavior alert with background
            (text_width, text_height), _ = cv2.getTextSize(behavior, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated_frame, (10, alert_y - text_height - 5), (10 + text_width, alert_y + 5), color, -1)
            cv2.putText(annotated_frame, behavior, (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            alert_y += 35
        
        # Draw emotion detections
        for emotion_data in emotions:
            x1, y1, x2, y2 = emotion_data["bbox"]
            emotion = emotion_data["emotion"]
            confidence = emotion_data["confidence"]
            is_threat = emotion_data["threat_emotion"]
            
            # Color based on threat level
            emotion_color = (0, 0, 255) if is_threat else (0, 255, 0)
            
            # Draw emotion box (smaller, at top of person box)
            emotion_y1 = max(0, y1 - 25)
            emotion_y2 = y1
            cv2.rectangle(annotated_frame, (x1, emotion_y1), (x2, emotion_y2), emotion_color, 2)
            
            # Draw emotion text
            emotion_text = f"{emotion}: {confidence:.1f}%"
            cv2.putText(annotated_frame, emotion_text, (x1, emotion_y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color, 1)
        
        return annotated_frame

    def process_frame(self, frame):
        """Main processing pipeline for a single frame"""
        try:
            # Detect objects
            raw_detections = self.detect_objects(frame)
            
            # Update tracker with detections
            stable_detections = self.object_tracker.update(raw_detections)
            
            # Detect poses
            poses = self.detect_poses(frame)
            
            # Analyze body language
            raw_behaviors = self.analyze_body_language(poses, stable_detections)
            
            # Stabilize behaviors
            stable_behaviors = self.behavior_stabilizer.update_behaviors(raw_behaviors)
            
            # Detect emotions
            person_detections = [d for d in stable_detections if d["category"] == "person"]
            raw_emotions = self.detect_emotions_ferplus(frame, person_detections)
            
            # Stabilize emotions
            stable_emotions = self.behavior_stabilizer.update_emotions(raw_emotions)
            
            # Calculate overall threat level
            overall_threat = self.calculate_overall_threat(stable_detections, stable_behaviors, stable_emotions)
            
            # Annotate frame
            annotated_frame = self.annotate_frame(frame, stable_detections, stable_behaviors, stable_emotions)
            
            # Add overall threat indicator
            annotated_frame = self.add_threat_indicator(annotated_frame, overall_threat)
            
            return annotated_frame, stable_detections, stable_behaviors, stable_emotions, overall_threat
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            return frame, [], [], [], 0

    def calculate_overall_threat(self, detections, behaviors, emotions):
        """Calculate overall threat level for the scene"""
        max_threat = 0
        
        # Get highest threat from objects
        for detection in detections:
            max_threat = max(max_threat, detection["threat_score"])
        
        # Add behavior threats
        for behavior in behaviors:
            if "[Score:" in behavior:
                try:
                    score_part = behavior.split("[Score:")[1].split("]")[0]
                    behavior_threat = int(score_part)
                    max_threat = max(max_threat, behavior_threat)
                except:
                    pass
        
        # Add emotion threats
        for emotion_data in emotions:
            if emotion_data["threat_emotion"] and emotion_data["confidence"] > 60:
                emotion_threat = min(int(emotion_data["confidence"]), 70)
                max_threat = max(max_threat, emotion_threat)
        
        return min(max_threat, 100)

    def add_threat_indicator(self, frame, overall_threat):
        """Add overall threat level indicator to frame"""
        height, width = frame.shape[:2]
        
        # Threat level indicator position (top right)
        indicator_x = width - 250
        indicator_y = 30
        
        # Get color and text
        color = get_threat_color(overall_threat)
        threat_text = get_threat_level_text(overall_threat)
        
        # Draw threat level box
        threat_display = f"THREAT LEVEL: {threat_text} ({overall_threat})"
        (text_width, text_height), _ = cv2.getTextSize(threat_display, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        # Draw background
        cv2.rectangle(frame, (indicator_x - 10, indicator_y - text_height - 10), 
                     (indicator_x + text_width + 10, indicator_y + 10), color, -1)
        
        # Draw border
        cv2.rectangle(frame, (indicator_x - 10, indicator_y - text_height - 10), 
                     (indicator_x + text_width + 10, indicator_y + 10), (255, 255, 255), 2)
        
        # Draw text
        cv2.putText(frame, threat_display, (indicator_x, indicator_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame

def get_user_input():
    """Interactive function to get user's choice of input source"""
    print("\n" + "="*60)
    print("    ENHANCED THREAT DETECTION SYSTEM")
    print("="*60)
    print("\nSelect input source:")
    print("1. Webcam (Real-time)")
    print("2. MP4 Video File")
    print("3. Exit")
    print("-" * 40)
    
    while True:
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == '1':
                return get_webcam_input()
            elif choice == '2':
                return get_video_file_input()
            elif choice == '3':
                print("Exiting...")
                return None, None, None
            else:
                print("❌ Invalid choice! Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return None, None, None

def get_webcam_input():
    """Get webcam configuration from user"""
    print("\n📹 WEBCAM CONFIGURATION")
    print("-" * 30)
    
    while True:
        try:
            cam_id = input("Enter webcam ID (0 for default, 1, 2... for other cameras): ").strip()
            if cam_id.isdigit():
                cam_id = int(cam_id)
                
                # Test if webcam exists
                test_cap = cv2.VideoCapture(cam_id)
                if test_cap.isOpened():
                    test_cap.release()
                    print(f"✅ Webcam {cam_id} detected successfully!")
                    
                    # Ask for output option
                    save_output = input("Do you want to save the output video? (y/n): ").strip().lower()
                    output_file = None
                    
                    if save_output in ['y', 'yes']:
                        output_file = input("Enter output filename (e.g., output.mp4): ").strip()
                        if not output_file.endswith('.mp4'):
                            output_file += '.mp4'
                        print(f"✅ Output will be saved as: {output_file}")
                    
                    return cam_id, True, output_file
                else:
                    print(f"❌ Webcam {cam_id} not found! Please try a different ID.")
            else:
                print("❌ Please enter a valid number!")
        except KeyboardInterrupt:
            print("\nReturning to main menu...")
            return None, None, None

def get_video_file_input():
    """Get video file configuration from user"""
    print("\n🎬 VIDEO FILE CONFIGURATION")
    print("-" * 35)
    
    while True:
        try:
            # Show available video files in current directory
            video_files = glob.glob("*.mp4") + glob.glob("*.avi") + glob.glob("*.mov") + glob.glob("*.mkv")
            
            if video_files:
                print("\n📁 Available video files in current directory:")
                for i, file in enumerate(video_files, 1):
                    print(f"   {i}. {file}")
                print(f"   {len(video_files) + 1}. Enter custom path")
                print("-" * 40)
                
                choice = input(f"Select a file (1-{len(video_files) + 1}) or enter custom path: ").strip()
                
                if choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(video_files):
                        video_path = video_files[choice_num - 1]
                    elif choice_num == len(video_files) + 1:
                        video_path = input("Enter full path to video file: ").strip()
                    else:
                        print("❌ Invalid selection!")
                        continue
                else:
                    video_path = choice
            else:
                print("📁 No video files found in current directory.")
                video_path = input("Enter full path to video file: ").strip()
            
            # Remove quotes if present
            video_path = video_path.strip('"').strip("'")
            
            # Check if file exists
            if os.path.exists(video_path):
                # Test if video file is valid
                test_cap = cv2.VideoCapture(video_path)
                if test_cap.isOpened():
                    # Get video info
                    fps = int(test_cap.get(cv2.CAP_PROP_FPS))
                    width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / fps if fps > 0 else 0
                    
                    test_cap.release()
                    
                    print(f"\n✅ Video file loaded successfully!")
                    print(f"   📝 File: {os.path.basename(video_path)}")
                    print(f"   📐 Resolution: {width}x{height}")
                    print(f"   ⏱️  Duration: {duration:.1f} seconds ({total_frames} frames)")
                    print(f"   🎯 FPS: {fps}")
                    
                    # Ask for output option
                    save_output = input("\nDo you want to save the processed output? (y/n): ").strip().lower()
                    output_file = None
                    
                    if save_output in ['y', 'yes']:
                        default_name = f"processed_{os.path.splitext(os.path.basename(video_path))[0]}.mp4"
                        output_file = input(f"Enter output filename (default: {default_name}): ").strip()
                        if not output_file:
                            output_file = default_name
                        elif not output_file.endswith('.mp4'):
                            output_file += '.mp4'
                        print(f"✅ Output will be saved as: {output_file}")
                    
                    return video_path, False, output_file
                else:
                    print("❌ Invalid video file! Cannot open for reading.")
            else:
                print(f"❌ File not found: {video_path}")
                print("Please check the path and try again.")
        except KeyboardInterrupt:
            print("\nReturning to main menu...")
            return None, None, None

def main():
    """Main function to run the enhanced threat detection system"""
    print("Loading Enhanced Threat Detection System...")
    
    # Get user input for source selection
    video_source, is_webcam, output_file = get_user_input()
    
    if video_source is None:
        return
    
    print("\n" + "="*50)
    print("INITIALIZING SYSTEM...")
    print("="*50)
    
    # Initialize detector
    detector = EnhancedThreatDetector()
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {video_source}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if not is_webcam else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else -1
    
    print(f"📐 Resolution: {width}x{height}")
    print(f"🎯 FPS: {fps}")
    if not is_webcam:
        duration = total_frames / fps if fps > 0 else 0
        print(f"⏱️  Duration: {duration:.1f} seconds ({total_frames} frames)")
    
    # Set camera properties for better performance (webcam only)
    if is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        width, height = 1280, 720
        print("📹 Webcam settings optimized to 1280x720@30fps")
    
    # Initialize video writer if output is specified
    video_writer = None
    if output_file:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        print(f"💾 Output will be saved to: {output_file}")
    
    print("\n✅ System initialized successfully!")
    print("\n🎮 CONTROLS:")
    print("   • Press 'q' to quit")
    print("   • Press 's' to save current frame")
    print("   • Press 'p' to pause/resume")
    print("   • Press 'r' to restart from beginning (video files only)")
    print("\n🚨 THREAT MONITORING ACTIVE...")
    print("=" * 50)
    
    frame_count = 0
    paused = False
    start_time = cv2.getTickCount()
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if is_webcam:
                        print("❌ Failed to grab frame from webcam")
                        break
                    else:
                        print("✅ End of video file reached")
                        break
                
                frame_count += 1
                
                # Process frame
                annotated_frame, detections, behaviors, emotions, overall_threat = detector.process_frame(frame)
                
                # Add frame counter and progress info
                info_text = f"Frame: {frame_count}"
                if not is_webcam and total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    info_text += f" | Progress: {progress:.1f}%"
                
                cv2.putText(annotated_frame, info_text, (10, annotated_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add input source info
                source_text = f"Source: {'Webcam' if is_webcam else 'Video File'}"
                cv2.putText(annotated_frame, source_text, (10, annotated_frame.shape[0] - 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Save to output video if specified
                if video_writer:
                    video_writer.write(annotated_frame)
                
                # Display frame
                cv2.imshow('Enhanced Threat Detection System', annotated_frame)
                
                # Print alerts for high threats
                if overall_threat >= 70:
                    timestamp = f"{frame_count:06d}" if not is_webcam else f"LIVE"
                    print(f"🚨 HIGH THREAT DETECTED! Level: {overall_threat} (Frame: {timestamp})")
                    if detections:
                        weapon_detections = [d for d in detections if "weapon" in d["category"]]
                        if weapon_detections:
                            for weapon in weapon_detections:
                                print(f"   🔫 {weapon['label']} detected (confidence: {weapon['confidence']:.2f})")
                    
                    if behaviors:
                        threatening_behaviors = [b for b in behaviors if "ARMED" in b or "Fighting" in b or "Aiming" in b]
                        if threatening_behaviors:
                            for behavior in threatening_behaviors:
                                print(f"   🥊 {behavior}")
                
                # Progress update for video files
                if not is_webcam and frame_count % 60 == 0:  # Every 60 frames (2 seconds)
                    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                    fps_actual = frame_count / elapsed_time
                    remaining_frames = total_frames - frame_count
                    eta_seconds = remaining_frames / fps_actual if fps_actual > 0 else 0
                    print(f"📊 Processing: {frame_count}/{total_frames} | "
                          f"Speed: {fps_actual:.1f} FPS | ETA: {eta_seconds:.0f}s")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                confirm = input("\n⚠️  Are you sure you want to quit? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    break
                else:
                    continue
            elif key == ord('s'):
                # Save current frame
                timestamp = int(cv2.getTickCount())
                filename = f"threat_frame_{frame_count}_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"💾 Frame saved as: {filename}")
            elif key == ord('p'):
                paused = not paused
                status = "⏸️ PAUSED" if paused else "▶️ RESUMED"
                print(f"{status}")
                if paused:
                    print("   Press 'p' again to resume or 'q' to quit")
            elif key == ord('r') and not is_webcam:
                # Restart video from beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                start_time = cv2.getTickCount()
                print("🔄 Video restarted from beginning")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user...")
    
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
            print(f"✅ Output video saved to: {output_file}")
        cv2.destroyAllWindows()
        
        # Final statistics
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print("\n" + "="*50)
        print("📊 PROCESSING COMPLETE")
        print("="*50)
        print(f"📈 Total frames processed: {frame_count}")
        print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
        print(f"🎯 Average processing speed: {avg_fps:.1f} FPS")
        
        if output_file and os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024*1024)  # MB
            print(f"💾 Output file size: {file_size:.1f} MB")
        
        print("🔒 Enhanced Threat Detection System stopped.")
        print("="*50)

if __name__ == "__main__":
    main()