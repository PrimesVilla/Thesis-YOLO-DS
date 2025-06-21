from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import serial.tools.list_ports
import hashlib
import imutils
import cv2
import time  # Import the time module

class WasteDetection:

    def __init__(self, capture, confidence_threshold, max_age, offset, line):
        self.capture = capture
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.names
        self.confidence_threshold = confidence_threshold
        self.line = line
        self.offset = offset
        self.bio_cls = [1, 5, 6, 11]
        self.recyclable_cls = [2, 3, 8, 10]
        self.special_cls = [0, 4, 7, 9]
        self.tracker = DeepSort(max_age=max_age)
        self.track_detections = {}

        # Initialize serial communication
        self.serialInst = serial.Serial()
        self.setup_serial()

    def setup_serial(self):
        ports = serial.tools.list_ports.comports()
        portsList = [str(one) for one in ports]
        for one in portsList:
            print(one)

        com = input("Select COM port for Arduino #: ")
        for i in range(len(portsList)):
            if portsList[i].startswith("COM" + str(com)):
                use = "COM" + str(com)
                print(use)
                self.serialInst.baudrate = 9600
                self.serialInst.port = use
                self.serialInst.open()


    def load_model(self):
            model = YOLO("best12cls.engine", task="detect")
            return model

    def predict(self, img):
        results = self.model.predict(source=img, conf=self.confidence_threshold, stream=True, device = 0)
        return results

    def plot_boxes(self, results, img):
        detections = []

        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = r
                detections.append(([int(x1), int(y1), (int(x2)-int(x1)), (int(y2)-int(y1))], conf, int(cls)))
        return detections, img

    def id_to_color(self, id_str):
        hash_object = hashlib.md5(id_str.encode())
        hash_hex = hash_object.hexdigest()
        r = int(hash_hex[:2], 16) % 256
        g = int(hash_hex[2:4], 16) % 256
        b = int(hash_hex[4:6], 16) % 256
        return r, g, b

    def line_detect(self, track_id, cls, cls_name):
        if track_id not in self.track_detections:
            if cls in self.bio_cls:
                print(cls, track_id, "detected.. moving BIO servo!")
                self.serialInst.write(str("BIO").encode('utf-8'))
            elif cls in self.recyclable_cls:
                print(cls, track_id, "detected.. moving RECYCLABLE servo!")
                self.serialInst.write(("REC").encode('utf-8'))
            elif cls in self.special_cls:
                print(cls, track_id, "detected.. moving SPECIAL servo!")
                self.serialInst.write(str("SPEC").encode('utf-8'))
            else:
                print("Detected class is not defined in parameters")
            self.track_detections[track_id] = cls_name


    def track_detect(self, detections, img):
        tracks = self.tracker.update_tracks(detections, frame=img)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            cls = self.CLASS_NAMES_DICT[track.det_class]
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            x3 = int(x1 + x2) // 2  # x coordinate for center point
            y3 = int(y1 + y2) // 2  # y coordinate for center point
            cv2.circle(img, (x3, y3), 3, (0, 255, 0), 2)
            if (self.line + self.offset) > x3 > (self.line - self.offset):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                self.line_detect(track_id, track.det_class, cls)

            cv2.putText(img, f'ID: {track_id} {cls} {track.det_conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)

        return img

    def __call__(self):
        cap = cv2.VideoCapture(self.capture)
        cap.set(3, 640)
        cap.set(4, 480)
        assert cap.isOpened()

        prev_time = 0  # Variable to store the time of the previous frame

        while True:
            ret, img = cap.read()

            if not ret:
                break

            # Measure FPS
            current_time = time.time()  # Get current time
            fps = 1 / (current_time - prev_time)  # Calculate FPS
            prev_time = current_time  # Update previous time

            results = self.predict(img)
            detections, frames = self.plot_boxes(results, img)
            detect_frame = self.track_detect(detections, frames)

            # Draw the red line
            cv2.line(img, (self.line + self.offset, 1), (self.line + self.offset, 480), (0, 0, 255), 2)
            cv2.line(img, (self.line - self.offset, 1), (self.line - self.offset, 480), (0, 0, 255), 2)
            # cv2.rectangle(img, (self.line-self.offset, 1), (self.line+self.offset, 480), (0, 0, 255), 2)

            # Display the FPS on the frame
            cv2.putText(detect_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            print(self.track_detections.items())

            cv2.imshow('Image', detect_frame)

            if cv2.waitKey(5) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Example usage:
detector = WasteDetection(capture=1, confidence_threshold=0.7, max_age=3, offset=10, line=320)
detector()
