import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#to control the volume usage
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol , maxVol , volBar, volPer= volRange[0] , volRange[1], 400, 0

#used to pop the webcam
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3,wCam)
cam.set(4,hCam)

#mediapipe code
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#initialize Windows Audio
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

#initialize Camera
cam = cv2.VideoCapture(0)

while cam.isOpened():
    success, image = cam.read()

    # Hand Landmark Detection
    with mp_hands.Hands(model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    lmList = []
    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[0]
        for id, lm in enumerate(myHand.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])

    # Check if Hand Landmarks are detected
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        # Draw Thumb and Index Finger
        cv2.circle(image, (x1, y1), 15, (255, 255, 255))
        cv2.circle(image, (x2, y2), 15, (255, 255, 255))
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Calculate distance between Thumb and Index Finger
        length = math.hypot(x2 - x1, y2 - y1)

        # Control Volume based on distance
        vol = np.interp(length, [50, 220], [minVol, maxVol])
        volume.SetMasterVolumeLevel(vol, None)
        volBar = np.interp(length, [50, 220], [400, 150])
        volPer = np.interp(length, [50, 220], [0, 100])

        # Draw Volume Bar
        cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
        cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
        cv2.putText(image, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_DUPLEX ,
                    1, (0, 0, 0), 3)

    # Display the image
    cv2.imshow('handDetector', image)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
