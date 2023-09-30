import cv2
import mediapipe as mp
import numpy as np
from math import ceil
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 650)
mp_pose = mp.solutions.pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
# For static images:
mp_model = mp_hands.Hands(
    static_image_mode=True, # only static images
    max_num_hands=1, # max 2 hands detection
    min_detection_confidence=0.5) # detection confidence


def wristAngles(image, points):
    #Loop through joint sets 
    a = points[0] # First coord
    b = points[1] # Second coord
    c = points[2] # Third coord
    
    radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = (radians*180.0/np.pi)
    angle = psMap(angle, 120, 240, 0, 180)
    angle = ceil(angle/10) * 10
    # angle -= 120
    # if angle > 180.0:
    #     angle = 360-angle
        
    cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [1280, 650]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return angle #image


def psMap(value, input_start, input_end, output_start, output_end):
    slope = 1.0 * (output_end - output_start) / (input_end - input_start)
    output = output_start + slope * (value - input_start)
    return output

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
    return angle #image

while cap.isOpened():
    _, image = cap.read()
    wA = 0
    tA = 0
    # now we flip image and convert to rgb image and input to model
    image = cv2.flip(image, 1)
    point_lists = []
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:

        # for hand_landmarks in results.multi_hand_landmarks:
        #     mp_drawing.draw_landmarks(
        #         image, # image to draw
        #         hand_landmarks, # model output
        #         mp_hands.HAND_CONNECTIONS, # hand connections
        #         mp_drawing_styles.get_default_hand_landmarks_style(),
        #         mp_drawing_styles.get_default_hand_connections_style())
        
        tA = fingerAngles(image, results, [[4,2,6]])
        
        for hand in results.multi_hand_landmarks:
            a = np.array([hand.landmark[4].x, hand.landmark[4].y])
            b = np.array([hand.landmark[0].x, hand.landmark[0].y])
            point_lists.append(a)
            point_lists.append(b)
            
    results = mp_pose.process(image)
    # if results.pose_landmarks:
    #     mp.solutions.drawing_utils.draw_landmarks(
    #     image,
    #     results.pose_landmarks,
    #     mp.solutions.pose.POSE_CONNECTIONS,
    #     mp.solutions.drawing_styles.get_default_pose_landmarks_style()
    # )
    a = results.pose_landmarks.landmark[13].x
    b = results.pose_landmarks.landmark[13].y
    point_lists.append(np.array([a,b]))
    
    if len(point_lists) == 3:
        wA = wristAngles(image, point_lists)

    if tA != 0:
        if tA >= 70:
            print("Accelerating")
        else:
            print("Decelrating")
    cv2.imshow("Detecting poses", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()