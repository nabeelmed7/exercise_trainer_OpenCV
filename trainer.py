import cv2
import mediapipe as mp
import time
import numpy as np
import math
import pose_module as pdn

detector = pdn.pose_detection()
count = 0
dir = 0

time_previous = 0
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = detector.pose_calculate(frame, False)
    lmList = detector.position_calculate(frame, False)

    if len(lmList) != 0:
        angle = detector.angle_calculate(frame, 12, 14, 16)
        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (220, 310), (650, 100))

        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        cv2.rectangle(frame, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(frame, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(frame, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_SIMPLEX, 4,
                    color, 4)
        
        cv2.rectangle(frame, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, str(int(count)), (45, 670), cv2.FONT_HERSHEY_SIMPLEX, 15,
                    (255, 0, 0), 25)
        
    time_current = time.time()
    fps = 1 / (time_current - time_previous)
    time_previous = time_current

    fps_text = "FPS: {:.1f}".format(fps)
    cv2.putText(frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    tracking_text = "Volume Control"
    cv2.putText(frame, tracking_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()