import cv2
import HandTrackingModule as htm
from pynput.mouse import Button, Controller
from screeninfo import get_monitors
import numpy as np

cap = cv2.VideoCapture(1) #Change this to 0 if not working


cap.set(3, 640) #Webcam Width
cap.set(4, 480) #Webcam Height



mouse = Controller()

smoothening = 5 

#Smothening values
plocX, plocY = 0,0
clocX, clocY = 0,0

#Frame reduction factor
frameR = 120

detector = htm.HandDetector()

def getScreenInfo(id):
    return [get_monitors()[id].width, get_monitors()[id].height]

screenSize = getScreenInfo(0)
camSize = [640, 480]

screenRatio = [int(screenSize[0] / 640), int(screenSize[1]/480)]
leftClicked = False
rightClicked = False

while True:
    _, frame = cap.read()
    frame = detector.findHands(frame)
    lmList, bbox = detector.findPosition(frame)
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        cv2.rectangle(frame, (frameR, frameR), ((camSize[0]- frameR), (camSize[1]-frameR)), (0, 255,0), 2)
        
        if fingers[2] == 0 and fingers[0]==0 and fingers[1] == 1: #Move mouse if only index up
            x3 = np.interp(x1, (frameR, camSize[0] - frameR), (0, screenSize[0])) #Reduce the frame size
            y3 = np.interp(y1, (frameR, camSize[1] - frameR), (0, screenSize[1]))

            
            clocX = plocX + (x3 - plocX) / smoothening #Smoothen the movement by discretizing the movement
            clocY = plocY + (y3 - plocY) / smoothening

            cv2.circle(frame, (x1,y1), 15, (0, 255, 0), cv2.FILLED)
            mouse.position = (screenSize[0] - clocX, clocY) #Flip along x axis

            plocX, plocY = clocX, clocY
        
        if fingers[2] == 1 and fingers[1] == 1: #Left click on index and middle finger touch
            length, frame, lineInfo = detector.findDistance(8, 12, frame)
            if length < 40:
                if leftClicked == False: #Prevent double clicking when fingers stay close
                    cv2.circle(frame,(lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    mouse.click(Button.left, 1)
                leftClicked = True
            else:
                leftClicked = False
        
        if fingers[0] == 1 and fingers[1] == 1: #Right click on thumb and index touch
            length, frame, lineInfo = detector.findDistance(4, 8, frame)
            if length < 40: #If the distance between the 2 fingers is smaller than 40
                if rightClicked == False: #Prevent double clicking when fingers stay close
                    cv2.circle(frame,(lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    mouse.click(Button.right, 1)
                rightClicked = True
            else:
                rightClicked = False

    cv2.imshow("frame", frame)
    cv2.waitKey(1)