import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
#Set Frame Size
cap.set(3, 1280)
cap.set(4, 650)
joint_list = [[8,7,6], [12,11,10], [16,15,14], [20,19,18],[2,3,4]]

def fingerAngles(image, results, joint_list):

# Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle
                
            cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [1280, 650]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return image
#Set the detection confidence and tracking confidence for better result
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
        # Rendering results
        if results.multi_hand_landmarks:
                      
            # Draw angles to image from joint list
            fingerAngles(image, results, [[4,2,6]])
        #Showing the camera
        cv2.imshow('Finger Angles', image)
        #exxit the program
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

# import cv2
# import mediapipe
 
# drawingModule = mediapipe.solutions.drawing_utils
# handsModule = mediapipe.solutions.hands
 
# capture = cv2.VideoCapture(0)
 
# frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
# frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
 
# with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
 
#     while (True):
 
#         ret, frame = capture.read()
 
#         results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
 
#         if results.multi_hand_landmarks != None:
#             for handLandmarks in results.multi_hand_landmarks:
#                 normalizedLandmark = handLandmarks.landmark[4]
#                 pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, frameWidth, frameHeight)
#                 cv2.circle(frame, pixelCoordinatesLandmark, 5, (0, 255, 0), -1)
#                 normalizedLandmark = handLandmarks.landmark[2]
#                 pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, frameWidth, frameHeight)
#                 cv2.circle(frame, pixelCoordinatesLandmark, 5, (0, 255, 0), -1)
 
 
#         cv2.imshow('Test hand', frame)
 
#         if cv2.waitKey(1) == 27:
#             break
 
# cv2.destroyAllWindows()
# capture.release()