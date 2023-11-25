import cv2
import mediapipe as mp
import numpy as np
from math import ceil, degrees

show_video = True

mp_drawing = mp.solutions.drawing_utils  # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore
mp_hands = mp.solutions.hands  # type: ignore

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 650)

mp_pose = mp.solutions.pose.Pose(  # type: ignore
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

mp_model = mp_hands.Hands(
    static_image_mode=False,  # only static images
    max_num_hands=1,  # max 1 hands detection
    min_detection_confidence=0.5,
)


def wristAngles(image, points):
    a = points[0]  # First coord
    b = points[1]  # Second coord
    c = points[2]  # Third coord

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = degrees(radians)  # need to map
    angle = psMap(angle, 120, 240, 0, 180)
    angle = ceil(angle / 10) * 10
    angle -= 90

    if angle < -90:
        if angle > -300:
            angle = -90
        elif angle < -300:
            angle = 90

    angle = int(round(angle, 2))

    if show_video:
        cv2.putText(
            image,
            str(round(angle, 2)),
            tuple(np.multiply(b, [1280, 650]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return angle, image


def fingerAngles(image, results, joint_list):
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        # Loop through joint sets
        for joint in joint_list:
            a = np.array(
                [hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]
            )  # First coord
            b = np.array(
                [hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]
            )  # Second coord
            c = np.array(
                [hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]
            )  # Third coord

            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
                a[1] - b[1], a[0] - b[0]
            )
            # angle = np.abs(radians * 180.0 / np.pi)

            angle = abs(degrees(radians))  # need to map
            if show_video:
                cv2.putText(
                    image,
                    str(round(angle, 2)),
                    tuple(np.multiply(b, [1280, 650]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
    return angle  # image


def psMap(value, input_start, input_end, output_start, output_end):
    slope = 1.0 * (output_end - output_start) / (input_end - input_start)
    output = output_start + slope * (value - input_start)
    return output


while cap.isOpened():
    _, image = cap.read()
    wrist_angle = 0
    thumb_angle = 0
    image = cv2.flip(image, 1)
    point_lists = []
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        thumb_angle = fingerAngles(image, results, [[6, 2, 4]])

        for hand in results.multi_hand_landmarks:
            a = np.array([hand.landmark[4].x, hand.landmark[4].y])
            b = np.array([hand.landmark[0].x, hand.landmark[0].y])
            point_lists.append(a)
            point_lists.append(b)

    results = mp_pose.process(image)

    point_lists.append(
        np.array(
            [
                results.pose_landmarks.landmark[13].x,
                results.pose_landmarks.landmark[13].y,
            ]
        )
    )

    if len(point_lists) == 3:
        wrist_angle, _ = wristAngles(image, point_lists)

    turn = 0
    print(wrist_angle)
    if wrist_angle >= 30:
        turn = 1
    elif wrist_angle <= -30:
        turn = -1
    # else:
    #     print(wrist_angle)

    data_to_send = {"turn_angle": turn, "accn": 1 if thumb_angle >= 70 else 0}

    print(data_to_send)

    if show_video:
        cv2.imshow("Detecting angles", image)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
