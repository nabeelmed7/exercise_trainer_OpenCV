import cv2
import mediapipe as mp
import time
import math
class pose_detection():
    def __init__(self, mode=False, model_complexity = 1, upperBody=False, smooth=True, confidence_detection = 0.7, confidence_tracking = 0.7):

        self.mode = mode
        self.model_complexity = model_complexity
        self.upBody = upperBody
        self.smooth = smooth
        self.confidence_detection = confidence_detection
        self.confidence_tracking = confidence_tracking
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.upBody, self.smooth, self.confidence_detection, self.confidence_tracking)
    def pose_calculate(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return frame
    def position_calculate(self, frame, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape

                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 0), cv2.FILLED)
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    h, w, c = frame.shape

                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), cv2.FILLED)
                    if id in [11, 12, 13, 14, 15, 16]:
                        cv2.line(frame, (self.lmList[id-1][1], self.lmList[id-1][2]), (cx, cy), (0, 255, 255), 3)
        return self.lmList
    def angle_calculate(self, frame, p1, p2, p3, draw=True):

        _, x1, y1 = self.lmList[p1]
        _, x2, y2 = self.lmList[p2]
        _, x3, y3 = self.lmList[p3]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle >  0:
            angle += 360

        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(frame, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(frame, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(frame, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(frame, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(frame, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle
def main():
    cap = cv2.VideoCapture(0)
    time_previous = 0
    detector = pose_detection()
    while True:
        success, frame = cap.read()
        frame = detector.pose_calculate(frame)
        lmList = detector.position_calculate(frame, draw=False)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(frame, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
        time_current = time.time()
        fps = 1 / (time_current - time_previous)
        time_previous = time_current
        fps_text = "FPS: {:.1f}".format(fps)
        cv2.putText(frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        tracking_text = "Pose Estimation"
        cv2.putText(frame, tracking_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()