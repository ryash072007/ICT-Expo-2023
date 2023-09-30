# Importing modules
import cv2
import mediapipe as mp
import numpy as np
from math import *

# Opening camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 650)

# Pose detector variable
mp_pose = mp.solutions.pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# For Debug
debug = True
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Taking input from camera and getting the pose 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive camera input. Exiting...")
        break
    
    # Getting the pose
    results = mp_pose.process(frame)
    
    # Drawing the pose
    if results.pose_landmarks:
        if debug:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_draw_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    cv2.imshow('Pose', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
