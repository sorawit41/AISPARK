from ultralytics import YOLO
import cv2
import math

# 1. Load the Model
print("Loading Model...")
model = YOLO('yolov8n.pt')

# 2. Open Webcam
# 0 is usually the default camera. If it doesn't work, try 1.
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Width
cap.set(4, 720)  # Height

print("Starting Camera... Press 'q' to exit.")

# 3. Class Names
classNames = model.names

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        break

    # 4. Perform Detection
    # stream=True is more efficient for video
    results = model(img, stream=True, verbose=False)

    # 5. Visualize (NO BOXES)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Center Point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Label Logic
            label_text = f"{currentClass}"
            color = (0, 255, 0) # Green

            if currentClass == "person":
                label_text = "Human" # Context: Human vs Ghost
                color = (0, 255, 255) # Yellow for humans

            # --- DRAWING (No Box, Just Text & Center) ---

            # Draw Center Dot
            cv2.circle(img, (cx, cy), 5, color, -1)
            
            # Draw Text
            text_display = f"{label_text} {int(conf*100)}%"
            
            # Put Text above the center
            cv2.putText(img, text_display, (cx - 20, cy - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 6. Display
    cv2.imshow("AI Live (No Box)", img)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
