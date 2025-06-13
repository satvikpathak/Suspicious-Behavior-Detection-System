import cv2
import numpy as np
from ultralytics import YOLO
import os
import logging
import math
from collections import defaultdict, deque

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Load YOLOv8 model for comprehensive object detection
object_model = YOLO("yolov8n.pt")  # Use yolov8s.pt or yolov8m.pt for better accuracy

# Load YOLOv8 model for pose estimation
pose_model = YOLO("yolov8n-pose.pt")

# Enhanced weapon categories
WEAPON_CLASSES = {
    'gun': ['gun', 'pistol', 'rifle', 'firearm'],
    'knife': ['knife', 'blade', 'sword', 'dagger'],
    'blunt_weapon': ['bat', 'stick', 'club', 'hammer'],
    'explosive': ['bomb', 'grenade', 'explosive']
}

# Suspicious objects that could be used as weapons
SUSPICIOUS_OBJECTS = ['bottle', 'scissors', 'axe', 'chainsaw', 'tool']

# Common objects for context
COMMON_OBJECTS = ['cell phone', 'phone', 'laptop', 'bag', 'backpack', 'book', 'cup', 'bottle']

# Threat levels
THREAT_LEVELS = {
    'HIGH': (0, 0, 255),      # Red
    'MEDIUM': (0, 165, 255),  # Orange
    'LOW': (0, 255, 255),     # Yellow
    'INFO': (255, 255, 0)     # Cyan
}

class StabilizedTracker:
    """Handles tracking and stabilization of detections"""
    
    def __init__(self, stability_threshold=5, max_age=15):
        self.stability_threshold = stability_threshold  # Frames needed to confirm detection
        self.max_age = max_age  # Max frames to keep detection without update
        self.tracked_objects = {}  # object_id: TrackedObject
        self.next_id = 0
        
    def update(self, detections):
        """Update tracked objects with new detections"""
        # Mark all current objects as not updated
        for obj in self.tracked_objects.values():
            obj.updated = False
        
        # Match new detections with existing tracked objects
        matched_detections = []
        for detection in detections:
            best_match_id = self._find_best_match(detection)
            
            if best_match_id is not None:
                # Update existing object
                self.tracked_objects[best_match_id].update(detection)
                matched_detections.append(best_match_id)
            else:
                # Create new tracked object
                new_id = self.next_id
                self.next_id += 1
                self.tracked_objects[new_id] = TrackedObject(new_id, detection)
                matched_detections.append(new_id)
        
        # Remove old objects that weren't updated
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
        
        # Calculate intersection
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
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
        self.threat_level = detection["threat_level"]
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
        
        # Update threat level with majority vote from recent detections
        if len(self.confidence_history) >= 3:
            # Use the most common threat level from recent detections
            self.threat_level = detection["threat_level"]  # For now, use latest
    
    def is_stable(self, threshold):
        """Check if object is stable enough to display"""
        return self.detection_count >= threshold
    
    def get_average_confidence(self):
        """Get average confidence from recent detections"""
        return sum(self.confidence_history) / len(self.confidence_history)
    
    def get_stable_bbox(self):
        """Get stabilized bounding box (average of recent positions)"""
        if len(self.bbox_history) == 1:
            return self.bbox_history[0]
        
        # Average the last few bounding boxes for stability
        recent_boxes = list(self.bbox_history)[-3:]  # Last 3 boxes
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
            "threat_level": self.threat_level
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
        self.stable_emotions = {}
        
    def update_behaviors(self, behaviors):
        """Update and stabilize behavior detections"""
        # Update behavior counts
        current_behaviors = set(behaviors)
        
        # Increment count for detected behaviors
        for behavior in current_behaviors:
            self.behavior_history[behavior] += 1
        
        # Decay count for non-detected behaviors
        for behavior in list(self.behavior_history.keys()):
            if behavior not in current_behaviors:
                self.behavior_history[behavior] = max(0, self.behavior_history[behavior] - 1)
        
        # Update active behaviors based on stability threshold
        new_active = set()
        for behavior, count in self.behavior_history.items():
            if count >= self.stability_frames:
                new_active.add(behavior)
                self.behavior_cooldown[behavior] = self.cooldown_frames
        
        # Handle cooldown for behaviors that are no longer detected
        for behavior in list(self.active_behaviors):
            if behavior not in new_active and self.behavior_cooldown[behavior] > 0:
                self.behavior_cooldown[behavior] -= 1
                if self.behavior_cooldown[behavior] > 0:
                    new_active.add(behavior)  # Keep showing during cooldown
        
        self.active_behaviors = new_active
        return list(self.active_behaviors)
    
    def update_emotions(self, emotions):
        """Update and stabilize emotion detections"""
        stable_emotions = []
        
        for emotion_data in emotions:
            person_key = f"{emotion_data['bbox'][0]}_{emotion_data['bbox'][1]}"  # Use position as key
            emotion = emotion_data['emotion']
            confidence = emotion_data['confidence']
            
            # Add to history
            self.emotion_history[person_key].append((emotion, confidence, emotion_data['threat_emotion']))
            
            # Calculate stable emotion for this person
            if len(self.emotion_history[person_key]) >= 5:  # Need at least 5 samples
                recent_emotions = list(self.emotion_history[person_key])[-5:]
                
                # Find most common emotion
                emotion_counts = defaultdict(int)
                threat_counts = defaultdict(int)
                confidence_sum = defaultdict(float)
                
                for emo, conf, threat in recent_emotions:
                    emotion_counts[emo] += 1
                    confidence_sum[emo] += conf
                    if threat:
                        threat_counts[emo] += 1
                
                # Get most frequent emotion
                most_common_emotion = max(emotion_counts, key=emotion_counts.get)
                avg_confidence = confidence_sum[most_common_emotion] / emotion_counts[most_common_emotion]
                is_threat = threat_counts[most_common_emotion] >= 2  # Majority of recent detections
                
                # Only show if confidence is high enough and emotion is consistent
                if emotion_counts[most_common_emotion] >= 3:  # Appeared in at least 3 of last 5 frames
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
        
        # Initialize stabilizers
        self.object_tracker = StabilizedTracker(stability_threshold=3, max_age=10)
        self.behavior_stabilizer = BehaviorStabilizer(stability_frames=5, cooldown_frames=10)
        
    def preprocess_frame(self, frame, target_size=(640, 640)):
        """Enhanced preprocessing with better quality retention"""
        height, width = frame.shape[:2]
        scale = min(target_size[0]/width, target_size[1]/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        y_offset = (target_size[1] - new_height) // 2
        x_offset = (target_size[0] - new_width) // 2
        padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return padded, scale, x_offset, y_offset

    def detect_objects(self, frame):
        """Enhanced object detection with detailed categorization"""
        try:
            results = object_model(frame, conf=0.25, iou=0.5)  # Lower confidence for better tracking
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf)
                        class_id = int(box.cls)
                        label = object_model.names[class_id]
                        
                        # Categorize detection
                        category = self.categorize_object(label)
                        threat_level = self.assess_threat_level(label, category)
                        
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "label": label,
                            "category": category,
                            "confidence": conf,
                            "threat_level": threat_level
                        })
            
            return detections
        except Exception as e:
            print(f"Error in detect_objects: {e}")
            return []

    def categorize_object(self, label):
        """Categorize detected objects"""
        label_lower = label.lower()
        
        # Check weapon categories
        for weapon_type, keywords in WEAPON_CLASSES.items():
            if any(keyword in label_lower for keyword in keywords):
                return f"weapon_{weapon_type}"
        
        # Check suspicious objects
        if any(obj in label_lower for obj in SUSPICIOUS_OBJECTS):
            return "suspicious_object"
        
        # Check common objects
        if any(obj in label_lower for obj in COMMON_OBJECTS):
            return "common_object"
        
        # Person detection
        if label_lower == "person":
            return "person"
        
        return "other"

    def assess_threat_level(self, label, category):
        """Assess threat level based on object type"""
        if "weapon_gun" in category or "weapon_explosive" in category:
            return "HIGH"
        elif "weapon_knife" in category or "weapon_blunt_weapon" in category:
            return "MEDIUM"
        elif category == "suspicious_object":
            return "LOW"
        else:
            return "INFO"

    def detect_poses(self, frame):
        """Enhanced pose detection with detailed keypoint analysis"""
        try:
            results = pose_model(frame, conf=0.4)  # Lower confidence for better tracking
            poses = []
            
            for result in results:
                if result.keypoints is not None:
                    for person_keypoints in result.keypoints.xy:
                        keypoints = person_keypoints.cpu().numpy()
                        if len(keypoints) >= 17:  # Ensure we have all keypoints
                            poses.append(keypoints)
            
            return poses
        except Exception as e:
            print(f"Error in detect_poses: {e}")
            return []

    def analyze_body_language(self, poses, detections):
        """Enhanced body language analysis"""
        behaviors = []
        person_detections = [d for d in detections if d["category"] == "person"]
        
        for i, pose in enumerate(poses):
            if i >= len(person_detections):
                break
                
            person = person_detections[i]
            pose_analysis = self.analyze_single_pose(pose)
            
            # Check for weapons near person
            weapon_nearby = self.check_weapons_nearby(person, detections)
            
            # Combine pose analysis with weapon detection
            for behavior in pose_analysis:
                if weapon_nearby and any(threat in behavior for threat in ["Aggressive", "Fighting", "Shooting"]):
                    behaviors.append(f"HIGH THREAT: {behavior} with weapon detected")
                else:
                    behaviors.append(behavior)
        
        return behaviors

    def analyze_single_pose(self, keypoints):
        """Analyze individual pose for threatening behaviors"""
        behaviors = []
        
        # Extract key body parts
        nose = keypoints[0]
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        # Check for valid keypoints (confidence > 0)
        valid_points = {
            'left_shoulder': left_shoulder[0] > 0,
            'right_shoulder': right_shoulder[0] > 0,
            'left_wrist': left_wrist[0] > 0,
            'right_wrist': right_wrist[0] > 0,
            'left_elbow': left_elbow[0] > 0,
            'right_elbow': right_elbow[0] > 0
        }
        
        # 1. Raised arms detection (hands up/surrender or aggressive)
        if valid_points['left_shoulder'] and valid_points['left_wrist']:
            if left_wrist[1] < left_shoulder[1] - 30:
                behaviors.append("ALERT: Left arm raised")
        
        if valid_points['right_shoulder'] and valid_points['right_wrist']:
            if right_wrist[1] < right_shoulder[1] - 30:
                behaviors.append("ALERT: Right arm raised")
        
        # 2. Fighting stance detection
        if all(valid_points[key] for key in ['left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist']):
            # Check if both hands are raised and forward (fighting position)
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            wrists_raised = (left_wrist[1] < shoulder_center_y + 20) and (right_wrist[1] < shoulder_center_y + 20)
            
            if wrists_raised:
                behaviors.append("THREAT: Fighting stance detected")
        
        # 3. Shooting stance detection
        if all(valid_points[key] for key in ['left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist', 'left_elbow', 'right_elbow']):
            # Extended arms in shooting position
            left_arm_extended = self.calculate_distance(left_shoulder, left_elbow) + self.calculate_distance(left_elbow, left_wrist)
            right_arm_extended = self.calculate_distance(right_shoulder, right_elbow) + self.calculate_distance(right_elbow, right_wrist)
            
            # Check if arms are extended forward
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            wrist_center_x = (left_wrist[0] + right_wrist[0]) / 2
            
            if wrist_center_x > shoulder_center_x + 40:  # Arms extended forward
                behaviors.append("HIGH THREAT: Shooting stance detected")
        
        # 4. Aggressive posture (wide stance, leaning forward)
        if left_ankle[0] > 0 and right_ankle[0] > 0 and left_hip[0] > 0 and right_hip[0] > 0:
            stance_width = abs(left_ankle[0] - right_ankle[0])
            hip_width = abs(left_hip[0] - right_hip[0])
            
            if stance_width > hip_width * 1.5:
                behaviors.append("ALERT: Wide aggressive stance")
        
        # 5. Lunging or attacking motion (forward lean)
        if nose[0] > 0 and left_hip[0] > 0 and right_hip[0] > 0:
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            if nose[0] > hip_center_x + 30:  # Head significantly forward of hips
                behaviors.append("THREAT: Lunging/attacking posture")
        
        return behaviors

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        if point1[0] > 0 and point1[1] > 0 and point2[0] > 0 and point2[1] > 0:
            return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        return 0

    def check_weapons_nearby(self, person, detections, threshold=150):
        """Check if weapons are near a person"""
        px1, py1, px2, py2 = person["bbox"]
        person_center = ((px1 + px2) // 2, (py1 + py2) // 2)
        
        weapon_detections = [d for d in detections if "weapon" in d["category"]]
        
        for weapon in weapon_detections:
            wx1, wy1, wx2, wy2 = weapon["bbox"]
            weapon_center = ((wx1 + wx2) // 2, (wy1 + wy2) // 2)
            distance = math.sqrt((person_center[0] - weapon_center[0])**2 + 
                               (person_center[1] - weapon_center[1])**2)
            
            if distance < threshold:
                return True
        
        return False

    def detect_emotions(self, frame, person_detections):
        """Basic emotion detection using facial analysis (DeepFace removed temporarily)"""
        emotions = []
        
        for person in person_detections:
            try:
                x1, y1, x2, y2 = person["bbox"]
                
                # Extract face region (upper third of person bbox)
                face_height = (y2 - y1) // 3
                face_y1 = y1
                face_y2 = y1 + face_height
                face_x1 = x1
                face_x2 = x2
                
                # Ensure face region is within frame bounds
                face_y1 = max(0, face_y1)
                face_y2 = min(frame.shape[0], face_y2)
                face_x1 = max(0, face_x1)
                face_x2 = min(frame.shape[1], face_x2)
                
                face_img = frame[face_y1:face_y2, face_x1:face_x2]
                
                if face_img.size == 0:
                    emotions.append({
                        "bbox": [x1, y1, x2, y2],
                        "emotion": "undetected",
                        "confidence": 0.0,
                        "threat_emotion": False
                    })
                    continue
                
                # Basic emotion estimation based on face brightness and contrast
                # This is a simplified approach until DeepFace is re-enabled
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray_face)
                contrast = np.std(gray_face)
                
                # Simple heuristic-based emotion estimation
                if contrast > 40 and brightness < 100:
                    emotion = "angry"
                    threat_emotion = True
                elif contrast < 20:
                    emotion = "neutral"
                    threat_emotion = False
                elif brightness > 150:
                    emotion = "happy"
                    threat_emotion = False
                else:
                    emotion = "neutral"
                    threat_emotion = False
                
                emotions.append({
                    "bbox": [x1, y1, x2, y2],
                    "emotion": emotion,
                    "confidence": 75.0,  # Placeholder confidence
                    "threat_emotion": threat_emotion
                })
                
            except Exception as e:
                print(f"Error in basic emotion detection: {e}")
                emotions.append({
                    "bbox": person["bbox"],
                    "emotion": "error",
                    "confidence": 0.0,
                    "threat_emotion": False
                })
        
        return emotions

    def annotate_frame(self, frame, detections, behaviors, emotions):
        """Enhanced frame annotation with better visualization and stability indicators"""
        annotated_frame = frame.copy()
        
        # Draw object detections with stability indicator
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            label = detection["label"]
            category = detection["category"]
            conf = detection["confidence"]
            threat_level = detection["threat_level"]
            
            # Choose color based on threat level
            color = THREAT_LEVELS.get(threat_level, (255, 255, 255))
            
            # Draw bounding box with thicker line for high threats
            thickness = 4 if threat_level in ["HIGH", "MEDIUM"] else 2
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            label_text = f"{label}: {conf:.2f}"
            if threat_level != "INFO":
                label_text = f"[{threat_level}] {label_text}"
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add stability indicator (small circle in top-right corner of bbox)
            stability_color = (0, 255, 0)  # Green for stable
            cv2.circle(annotated_frame, (x2 - 10, y1 + 10), 5, stability_color, -1)
        
        # Draw behavior alerts with better spacing
        alert_y = 30
        for behavior in behaviors:
            if "HIGH THREAT" in behavior:
                color = THREAT_LEVELS["HIGH"]
                prefix = "🔴 "
            elif "THREAT" in behavior:
                color = THREAT_LEVELS["MEDIUM"]
                prefix = "🟠 "
            elif "ALERT" in behavior:
                color = THREAT_LEVELS["LOW"]
                prefix = "🟡 "
            else:
                color = THREAT_LEVELS["INFO"]
                prefix = "🔵 "
            
            # Add background for better readability
            text = f"{prefix}{behavior}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated_frame, (5, alert_y - text_height - 5), (15 + text_width, alert_y + 5), (0, 0, 0), -1)
            
            cv2.putText(annotated_frame, text, (10, alert_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            alert_y += 30
        
        # Draw emotions with better formatting
        for emotion_data in emotions:
            x1, y1, x2, y2 = emotion_data["bbox"]
            emotion = emotion_data["emotion"]
            confidence = emotion_data["confidence"]
            is_threat = emotion_data["threat_emotion"]
            
            emotion_color = THREAT_LEVELS["MEDIUM"] if is_threat else THREAT_LEVELS["INFO"]
            
            if is_threat:
                emotion_text = f"⚠️ THREAT: {emotion} ({confidence:.1f}%)"
            else:
                emotion_text = f"😐 {emotion} ({confidence:.1f}%)"
            
            # Add background for emotion text
            (text_width, text_height), _ = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y2 + 15), (x1 + text_width, y2 + 15 + text_height + 5), (0, 0, 0), -1)
            
            cv2.putText(annotated_frame, emotion_text, (x1, y2 + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)
        
        # Add system status
        status_text = f"Tracking: {len(detections)} objects | Behaviors: {len(behaviors)} | Stable Detection"
        cv2.rectangle(annotated_frame, (10, annotated_frame.shape[0] - 40), 
                     (len(status_text) * 12, annotated_frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.putText(annotated_frame, status_text, (15, annotated_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return annotated_frame

    def process_frame(self, frame):
        """Main frame processing pipeline with stabilization"""
        # Detect objects
        raw_detections = self.detect_objects(frame)
        
        # Stabilize object detections using tracker
        stable_detections = self.object_tracker.update(raw_detections)
        
        # Detect poses
        poses = self.detect_poses(frame)
        
        # Analyze body language and behaviors
        raw_behaviors = self.analyze_body_language(poses, stable_detections)
        
        # Stabilize behaviors
        stable_behaviors = self.behavior_stabilizer.update_behaviors(raw_behaviors)
        
        # Get person detections for emotion analysis
        person_detections = [d for d in stable_detections if d["category"] == "person"]
        
        # Detect emotions
        raw_emotions = self.detect_emotions(frame, person_detections)
        
        # Stabilize emotions
        stable_emotions = self.behavior_stabilizer.update_emotions(raw_emotions)
        
        # Store frame history for temporal analysis
        self.frame_history.append({
            'frame': frame.copy(),
            'detections': stable_detections,
            'behaviors': stable_behaviors,
            'emotions': stable_emotions
        })
        
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
        
        # Annotate frame
        annotated_frame = self.annotate_frame(frame, stable_detections, stable_behaviors, stable_emotions)
        
        return annotated_frame, {
            'detections': stable_detections,
            'behaviors': stable_behaviors,
            'emotions': stable_emotions,
            'threat_level': self.calculate_overall_threat_level(stable_detections, stable_behaviors, stable_emotions)
        }
    
    def calculate_overall_threat_level(self, detections, behaviors, emotions):
        """Calculate overall threat level for the frame"""
        threat_score = 0
        
        # Score based on detections
        for detection in detections:
            if detection["threat_level"] == "HIGH":
                threat_score += 10
            elif detection["threat_level"] == "MEDIUM":
                threat_score += 5
            elif detection["threat_level"] == "LOW":
                threat_score += 2
        
        # Score based on behaviors
        for behavior in behaviors:
            if "HIGH THREAT" in behavior:
                threat_score += 15
            elif "THREAT" in behavior:
                threat_score += 8
            elif "ALERT" in behavior:
                threat_score += 3
        
        # Score based on emotions
        for emotion in emotions:
            if emotion["threat_emotion"]:
                threat_score += 5
        
        # Determine overall threat level
        if threat_score >= 20:
            return "CRITICAL"
        elif threat_score >= 10:
            return "HIGH"
        elif threat_score >= 5:
            return "MEDIUM"
        elif threat_score > 0:
            return "LOW"
        else:
            return "NORMAL"
    
    def get_threat_summary(self):
        """Get summary of threats from recent frames"""
        if not self.frame_history:
            return "No data available"
        
        recent_frames = self.frame_history[-5:]  # Last 5 frames
        
        all_detections = []
        all_behaviors = []
        all_emotions = []
        
        for frame_data in recent_frames:
            all_detections.extend(frame_data['detections'])
            all_behaviors.extend(frame_data['behaviors'])
            all_emotions.extend(frame_data['emotions'])
        
        # Count threat types
        threat_counts = defaultdict(int)
        for detection in all_detections:
            if detection["threat_level"] != "INFO":
                threat_counts[detection["threat_level"]] += 1
        
        behavior_threats = len([b for b in all_behaviors if "THREAT" in b or "ALERT" in b])
        emotion_threats = len([e for e in all_emotions if e["threat_emotion"]])
        
        summary = f"Recent Activity (last 5 frames):\n"
        summary += f"- High threats: {threat_counts['HIGH']}\n"
        summary += f"- Medium threats: {threat_counts['MEDIUM']}\n"
        summary += f"- Low threats: {threat_counts['LOW']}\n"
        summary += f"- Threatening behaviors: {behavior_threats}\n"
        summary += f"- Threatening emotions: {emotion_threats}"
        
        return summary


def main():
    """Main function to run the enhanced threat detection system"""
    print("Initializing Enhanced Threat Detection System...")
    
    # Initialize the detector
    detector = EnhancedThreatDetector()
    
    # Initialize video capture (0 for webcam, or path to video file)
    cap = cv2.VideoCapture(0)  # Change to video file path if needed
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Set video properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Performance tracking
    frame_count = 0
    fps_start_time = cv2.getTickCount()
    
    print("Starting threat detection... Press 'q' to quit, 's' for summary")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame")
                break
            
            # Process frame
            annotated_frame, results = detector.process_frame(frame)
            
            # Add FPS counter
            frame_count += 1
            if frame_count % 30 == 0:  # Calculate FPS every 30 frames
                fps_end_time = cv2.getTickCount()
                fps = 30 / ((fps_end_time - fps_start_time) / cv2.getTickFrequency())
                fps_start_time = fps_end_time
                
                # Display FPS on frame
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                           (annotated_frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add overall threat level indicator
            threat_level = results['threat_level']
            threat_color = {
                'CRITICAL': (0, 0, 255),  # Red
                'HIGH': (0, 69, 255),     # Orange Red
                'MEDIUM': (0, 165, 255),  # Orange
                'LOW': (0, 255, 255),     # Yellow
                'NORMAL': (0, 255, 0)     # Green
            }.get(threat_level, (255, 255, 255))
            
            # Draw threat level indicator
            cv2.rectangle(annotated_frame, (annotated_frame.shape[1] - 200, 50), 
                         (annotated_frame.shape[1] - 10, 90), threat_color, -1)
            cv2.putText(annotated_frame, f"THREAT: {threat_level}", 
                       (annotated_frame.shape[1] - 190, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Enhanced Threat Detection System', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                summary = detector.get_threat_summary()
                print("\n" + "="*50)
                print("THREAT SUMMARY")
                print("="*50)
                print(summary)
                print("="*50 + "\n")
            elif key == ord('r'):
                # Reset tracking history
                detector.object_tracker = StabilizedTracker(stability_threshold=3, max_age=10)
                detector.behavior_stabilizer = BehaviorStabilizer(stability_frames=5, cooldown_frames=10)
                detector.frame_history = []
                print("Tracking history reset")
            elif key == ord('h'):
                # Display help
                print("\n" + "="*40)
                print("KEYBOARD CONTROLS")
                print("="*40)
                print("q - Quit application")
                print("s - Show threat summary")
                print("r - Reset tracking history")
                print("h - Show this help")
                print("="*40 + "\n")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"Error during processing: {e}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Enhanced Threat Detection System stopped")


def test_on_image(image_path):
    """Test the system on a single image"""
    print(f"Testing on image: {image_path}")
    
    detector = EnhancedThreatDetector()
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Process image
    annotated_frame, results = detector.process_frame(frame)
    
    # Display results
    print(f"Overall threat level: {results['threat_level']}")
    print(f"Detections: {len(results['detections'])}")
    print(f"Behaviors: {len(results['behaviors'])}")
    print(f"Emotions: {len(results['emotions'])}")
    
    # Show image
    cv2.imshow('Threat Analysis Result', annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_on_video(video_path):
    """Test the system on a video file"""
    print(f"Testing on video: {video_path}")
    
    detector = EnhancedThreatDetector()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = 0
    threat_log = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        annotated_frame, results = detector.process_frame(frame)
        
        # Log threats
        if results['threat_level'] != 'NORMAL':
            threat_log.append({
                'frame': frame_count,
                'threat_level': results['threat_level'],
                'detections': len(results['detections']),
                'behaviors': len(results['behaviors'])
            })
        
        # Display every 10th frame for performance
        if frame_count % 10 == 0:
            cv2.imshow('Video Threat Analysis', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print(f"\nVideo Analysis Complete:")
    print(f"Total frames processed: {frame_count}")
    print(f"Threat incidents: {len(threat_log)}")
    
    if threat_log:
        print("\nThreat Log:")
        for incident in threat_log:
            print(f"Frame {incident['frame']}: {incident['threat_level']} threat "
                  f"({incident['detections']} objects, {incident['behaviors']} behaviors)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Threat Detection System')
    parser.add_argument('--mode', choices=['live', 'image', 'video'], default='live',
                       help='Detection mode: live webcam, single image, or video file')
    parser.add_argument('--source', type=str, help='Path to image or video file')
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='Detection confidence threshold (0.0-1.0)')
    
    args = parser.parse_args()
    
    if args.mode == 'live':
        main()
    elif args.mode == 'image':
        if args.source:
            test_on_image(args.source)
        else:
            print("Error: Please provide image path with --source")
    elif args.mode == 'video':
        if args.source:
            test_on_video(args.source)
        else:
            print("Error: Please provide video path with --source")
    else:
        print("Invalid mode selected")