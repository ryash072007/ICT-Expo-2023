import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
#Set Frame Size
cap.set(3, 1280)
cap.set(4, 650)
mp_pose = mp.solutions.pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = mp_pose.process(image)
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
            image,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )
        cv2.imshow("Detecting poses", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

        
