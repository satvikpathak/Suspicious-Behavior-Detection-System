from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import csv
import os
import time
import cv2

def send_alert(behavior, frame_id, frame, face_data):
    print(f"ALERT: {behavior} at frame {frame_id}")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs('demo', exist_ok=True)
    
    # PDF report
    pdf_path = f"demo/report_{timestamp}_{frame_id}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 750, "Suspicious Behavior Report")
    c.drawString(100, 730, f"Behavior: {behavior}")
    c.drawString(100, 710, f"Frame: {frame_id}")
    c.drawString(100, 690, f"Time: {timestamp}")
    if face_data:
        faces = [f"{data['name']} ({data['emotion']})" for data in face_data.values()]
        c.drawString(100, 670, f"Faces: {', '.join(faces)}")
    c.save()
    
    # CSV log for suspicious faces
    if "emotion" in behavior.lower():
        with open('demo/suspicious_faces.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['Timestamp', 'Frame', 'Behavior'])
            writer.writerow([timestamp, frame_id, behavior])
    
    cv2.imwrite(f"demo/frame_{timestamp}_{frame_id}.jpg", frame)