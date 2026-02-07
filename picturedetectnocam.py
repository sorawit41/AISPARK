from ultralytics import YOLO
import cv2
import math
import os

# 1. Load the Model
# We use YOLOv8 Nano for speed. It will download if not present.
print("Loading Model...")
model = YOLO('yolov8n.pt')

# 2. Load the Image
img_path = "test.jpg"  # Default file to look for

if not os.path.exists(img_path):
    print(f"Error: Could not find '{img_path}' in the current directory.")
    print("Please place an image named 'test.jpg' in this folder and try again.")
    exit()

img = cv2.imread(img_path)

# 3. Perform Detection
print("Detecting objects...")
results = model(img)

# 4. Class Names (COCO)
classNames = model.names

# 5. Process Results & Visualize (NO BOXES)
for r in results:
    boxes = r.boxes
    for box in boxes:
        # Get coordinates to find the center
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Calculate Center Point
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Get Confidence
        conf = math.ceil((box.conf[0] * 100)) / 100
        
        # Get Class Name
        cls = int(box.cls[0])
        currentClass = classNames[cls]

        # LOGIC: "Person" -> "Human"
        # Everything else is just its name. 
        # "Ghost" is not a standard class, but we could pretend low confidence is a ghost? 
        # For now, let's keep it accurate to the AI model.
        
        label_text = f"{currentClass}"
        color = (0, 255, 0) # Green

        if currentClass == "person":
            label_text = f"Human"
            color = (0, 255, 255) # Yellow for humans

        # --- DRAWING (No Box, Just Text & Center) ---
        
        # 1. Draw a small circle at the center
        cv2.circle(img, (cx, cy), 5, color, -1)
        
        # 2. Prepare Text
        text_display = f"{label_text} {int(conf*100)}%"
        
        # 3. Put Text above the center
        # font, scale, thickness
        cv2.putText(img, text_display, (cx - 20, cy - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# 6. Show and Save Result
print("Done! Saving result to 'result.jpg'...")
cv2.imwrite("result.jpg", img)

print("Displaying result. Press any key to close the window.")
cv2.imshow("AI Object Detection (No Box)", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
