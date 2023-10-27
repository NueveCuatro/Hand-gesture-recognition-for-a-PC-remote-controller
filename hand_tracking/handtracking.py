import cv2
import mediapipe as mp
import time
import HandTrackingModule as htModule


pTime = 0
cTime = 0
cap = cv2.VideoCapture(1)
detector = htModule.HandDetector()
while True:
    ret, frame = cap.read()
    frame = detector.findHands(frame)
    position = detector.findPosition(frame)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)


    cv2.imshow('frame', frame)
    cv2.waitKey(1)