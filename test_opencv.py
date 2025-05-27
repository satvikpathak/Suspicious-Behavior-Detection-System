# test_opencv.py
import cv2
print("OpenCV version:", cv2.__version__)
cap = cv2.VideoCapture(0)
print("Webcam accessible:", cap.isOpened())
cap.release()