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
    min_tracking_confidence=0.5
)

# For Debug
debug = True
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Angle Function
def _angle(a, b, c, image):
    origin = np.array([b.x, b.y, b.z])
    first = np.array([a.x, a.y, a.z])
    second = np.array([c.x, c.y, c.z])
    vecA = first - origin
    vecB = second - origin
    dot_product = np.dot(vecA, vecB)
    angle = np.arccos(dot_product / (np.linalg.norm(vecA) * np.linalg.norm(vecB)))
    angle = np.rad2deg(angle)
    cv2.putText(image, str(round(angle, 2)), tuple(np.multiply([b.x, b.y], [1280, 650]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    return image


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
                frame,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
        
        # Getting the landmarks
        landmarks = results.pose_landmarks.landmark
        frame = _angle(landmarks[19], landmarks[15], landmarks[13], frame)
        

    cv2.imshow("Pose", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
