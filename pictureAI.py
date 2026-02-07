from ultralytics import YOLO
import cv2
import math

# Load the YOLOv8 model (Nano version for speed)
# It will download 'yolov8n.pt' automatically on first run
model = YOLO('yolov8n.pt')

# Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Width
cap.set(4, 720)  # Height

# Class names for YOLOv8 (COCO dataset)
classNames = ["Person", "Bicycle", "Car", "Motorbike", "Aeroplane", "Bus", "Train", "Truck", "Boat",
              "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench", "Bird", "Cat",
              "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella",
              "Handbag", "Tie", "Suitcase", "Frisbee", "Skis", "Snowboard", "Sports Ball", "Kite", "Baseball Bat",
              "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket", "Bottle", "Wine Glass", "Cup",
              "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple", "Sandwich", "Orange", "Broccoli",
              "Carrot", "Hot Dog", "Pizza", "Donut", "Cake", "Chair", "Sofa", "Pottedplant", "Bed",
              "Diningtable", "Toilet", "Tvmonitor", "Laptop", "Mouse", "Remote", "Keyboard", "Cell Phone",
              "Microwave", "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase", "Scissors",
              "Teddy Bear", "Hair Drier", "Toothbrush"
              ]

print("Starting Camera... Press 'q' to exit.")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        break

    # Run YOLO detection
    results = model(img, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Color setup
            if currentClass == "Person":
                # Green for humans
                color = (0, 255, 0) 
                label = f"Human ({conf})"
            else:
                # Red for objects
                color = (0, 0, 255)
                label = f"{currentClass} ({conf})"

            # Draw Box & Text
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # Label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img, (x1, y1 - 30), (x1 + w, y1), color, -1)
            
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the image
    cv2.imshow("Image Analysis (Human vs Object)", img)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
