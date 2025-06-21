from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import hashlib
import cv2
import serial.tools.list_ports

class WasteDetection:

    def __init__(self, capture, confidence_threshold=0.5, max_age=3, offset=8, line=200):
        self.capture = capture
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.names
        self.confidence_threshold = confidence_threshold
        self.line = line
        self.offset = offset
        self.bio_cls = [1,7,8,10,14]
        self.recyclable_cls = [2,3,4,11,13]
        self.special_cls = [0,5,6,9,12]
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
        model = YOLO("best.pt")
        model.fuse()
        return model

    def predict(self, img):
        results = self.model(img, stream=True)
        return results

    def plot_boxes(self, results, img):
        detections = []

        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, conf, current_class = r
                x1 = float(x1)
                x2 = float(x2)
                y1 = float(y1)
                y2 = float(y2)
                current_class = int(current_class)
                if conf > self.confidence_threshold:
                    detections.append(([x1, y1, float(x2 - x1), float(y2 - y1)], conf, current_class))
        return detections, img

    def id_to_color(self, id_str):
        hash_object = hashlib.md5(id_str.encode())
        hash_hex = hash_object.hexdigest()
        r = int(hash_hex[:2], 16) % 256
        g = int(hash_hex[2:4], 16) % 256
        b = int(hash_hex[4:6], 16) % 256
        return (r, g, b)

    def line_detect(self, track_id, cls):
        if cls in self.bio_cls:
            print(cls, track_id, "detected.. moving BIO servo!")
            #self.serialInst.write(str("BIO").encode('utf-8'))
        elif cls in self.recyclable_cls:
            print(cls, track_id, "detected.. moving RECYCLABLE servo!")
            #self.serialInst.write(str("REC").encode('utf-8'))
        elif cls in self.special_cls:
            print(cls, track_id, "detected.. moving SPECIAL servo!")
            #self.serialInst.write(str("SPEC").encode('utf-8'))
        else:
            print("Detected class is not defined in parameters")

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
            color = self.id_to_color(str(track_id))
            cv2.circle(img, (x3, y3), 2, (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            if (self.line + self.offset) > x3 > (self.line - self.offset):
                self.line_detect(track_id, track.det_class)
                self.serialInst.write("REC".encode('utf-8'))
                self.track_detections[track_id] = cls

            cv2.putText(img, f'ID:{track_id} {cls}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)

        return img

    def __call__(self):
        cap = cv2.VideoCapture(self.capture)
        assert cap.isOpened()

        while True:
            ret, img = cap.read()

            if not ret:
                break

            results = self.predict(img)
            detections, frames = self.plot_boxes(results, img)
            detect_frame = self.track_detect(detections, frames)
            cv2.line(img, (self.line, 1), (self.line, 480), (0, 255, 0), 2)
            print(self.track_detections.items())

            cv2.imshow('Image', detect_frame)
            if cv2.waitKey(5) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Example usage:
detector = WasteDetection(capture=0)
detector()
