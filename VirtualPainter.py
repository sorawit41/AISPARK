import cv2
import numpy as np
import os
import HandTrackingModule as htm

def main():
    #######################
    brushThickness = 15
    eraserThickness = 50
    #######################

    # Setup Setup
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = htm.HandDetector(detectionCon=0.85)

    # UI Variables
    drawColor = (255, 0, 255) # Default Color (Pink)
    
    # Canvas
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    # Previous coordinates
    xp, yp = 0, 0

    while True:
        # 1. Import image
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)

        # 2. Find Hand Landmarks
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # Tip of Index and Middle Fingers
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            # 3. Check which fingers are up
            fingers = detector.fingersUp()
            
            # 4. If Selection Mode - Two fingers are up
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0 # Reset previous points
                print("Selection Mode")
                
                # Check for click in header area (Selection Logic)
                if y1 < 125:
                    if 250 < x1 < 450:
                        drawColor = (255, 0, 255) # Pink
                    elif 550 < x1 < 750:
                        drawColor = (255, 0, 0) # Blue
                    elif 800 < x1 < 950:
                        drawColor = (0, 255, 0) # Green
                    elif 1050 < x1 < 1200:
                        drawColor = (0, 0, 0) # Eraser

                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            # 5. If Drawing Mode - Index finger is up
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                print("Drawing Mode")
                
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if drawColor == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                
                xp, yp = x1, y1

        # Converting canvas to gray to create a mask
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        
        # Combine image and canvas
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        # Draw UI Header (Simple rectangles for this demo instead of images)
        # Colors: Pink, Blue, Green, Eraser(Black/White for visibility in header)
        cv2.rectangle(img, (0, 0), (1280, 125), (50, 50, 50), cv2.FILLED) # Header Background
        cv2.rectangle(img, (250, 10), (450, 115), (255, 0, 255), cv2.FILLED) # Pink
        cv2.rectangle(img, (550, 10), (750, 115), (255, 0, 0), cv2.FILLED) # Blue
        cv2.rectangle(img, (800, 10), (950, 115), (0, 255, 0), cv2.FILLED) # Green
        cv2.rectangle(img, (1050, 10), (1200, 115), (255, 255, 255), cv2.FILLED) # Eraser Visual
        cv2.putText(img, "Eraser", (1080, 75), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)


        cv2.imshow("Image", img)
        # cv2.imshow("Canvas", imgCanvas)
        # cv2.imshow("Inv", imgInv)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
