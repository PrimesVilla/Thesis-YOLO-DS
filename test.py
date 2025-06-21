from ultralytics import YOLO
import cv2
import imutils

# Load the YOLOv8 model
model = YOLO("best.pt")  # or "yolov8s.pt", "yolov8m.pt", etc., depending on your needs

# Replace with your ESP32-CAM stream URL
stream_url = "http://192.168.1.13:81/stream"
cap = cv2.VideoCapture(stream_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = imutils.resize(frame, width=640)

    # Run YOLOv8 inference on the frame
    results = model(img, conf=0.5)

    # Visualize the results
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
