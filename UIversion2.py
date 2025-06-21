import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import serial.tools.list_ports
import hashlib
import cv2
import time

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

        # Waste counters
        self.bio_count = 0
        self.recyclable_count = 0
        self.special_count = 0

        # Initialize serial communication
        self.serialInst = serial.Serial()
        self.setup_serial()

        # GUI Setup
        self.root = tk.Tk()
        self.root.title("Waste Detection System")
        self.video_label = tk.Label(self.root)
        self.video_label.grid(row=0, column=0, columnspan=3)

        # Waste Count Labels
        self.bio_label = tk.Label(self.root, text=f'Bio: {self.bio_count}', font=("Helvetica", 16))
        self.bio_label.grid(row=1, column=0)
        self.recyclable_label = tk.Label(self.root, text=f'Recyclable: {self.recyclable_count}', font=("Helvetica", 16))
        self.recyclable_label.grid(row=1, column=1)
        self.special_label = tk.Label(self.root, text=f'E-waste: {self.special_count}', font=("Helvetica", 16))
        self.special_label.grid(row=1, column=2)

        # Text widget for displaying detected items
        self.detect_text = tk.Text(self.root, height=10, width=75)
        self.detect_text.grid(row=2, column=0, columnspan=3)

        self.root.bind('<KeyPress>', self.keypress)

    def keypress(self, event):
        if event.char == 'q':
            print("Q key pressed, quitting application...")
            self.root.quit()

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
        model = YOLO("best12cls.pt", task="detect")
        return model

    def predict(self, img):
        results = self.model.predict(source=img, conf=self.confidence_threshold, stream=True, device=0)
        return results

    def plot_boxes(self, results, img):
        detections = []

        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = r
                detections.append(([int(x1), int(y1), (int(x2) - int(x1)), (int(y2) - int(y1))], conf, int(cls)))
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
            # Insert detected item into the text widget
            detection_text = f"Track ID: {track_id}, Class: {cls_name}\n"
            self.detect_text.insert(tk.END, detection_text)

            if cls in self.bio_cls:
                print(cls, track_id, "detected.. moving BIO servo!")
                self.serialInst.write(str("BIO").encode('utf-8'))
                self.bio_count += 1
                self.update_gui()
            elif cls in self.recyclable_cls:
                print(cls, track_id, "detected.. moving RECYCLABLE servo!")
                self.serialInst.write(("REC").encode('utf-8'))
                self.recyclable_count += 1
                self.update_gui()
            elif cls in self.special_cls:
                print(cls, track_id, "detected.. moving SPECIAL servo!")
                self.serialInst.write(str("SPEC").encode('utf-8'))
                self.special_count += 1
                self.update_gui()
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
            x3 = int(x1 + x2) // 2
            y3 = int(y1 + y2) // 2
            cv2.circle(img, (x3, y3), 3, (0, 255, 0), 2)
            if (self.line + self.offset) > x3 > (self.line - self.offset):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                self.line_detect(track_id, track.det_class, cls)

            cv2.putText(img, f'ID: {track_id} {cls} {track.det_conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)

        return img

    def update_gui(self):
        self.bio_label.config(text=f'Bio: {self.bio_count}')
        self.recyclable_label.config(text=f'Recyclable: {self.recyclable_count}')
        self.special_label.config(text=f'E-waste: {self.special_count}')

    def video_loop(self):
        cap = cv2.VideoCapture(self.capture)
        cap.set(3, 640)
        cap.set(4, 480)
        assert cap.isOpened()

        prev_time = 0
        frame_count = 0

        while True:
            ret, img = cap.read()

            if not ret:
                break

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            results = self.predict(img)
            detections, frames = self.plot_boxes(results, img)
            detect_frame = self.track_detect(detections, frames)

            # Draw the red line
            cv2.line(img, (self.line + self.offset, 1), (self.line + self.offset, 480), (0, 0, 255), 2)
            cv2.line(img, (self.line - self.offset, 1), (self.line - self.offset, 480), (0, 0, 255), 2)

            cv2.putText(detect_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Convert the frame to a format suitable for tkinter
            img_rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            # Update the video label with the new frame
            self.video_label.config(image=img_tk)
            self.video_label.image = img_tk

            key = cv2.waitKey(1) & 0xFF

            if key == ord('w'):  # If 'w' key is pressed
                frame_count += 1
                filename = f'captured_frame_{frame_count}.jpg'
                cv2.imwrite(filename, img)
                print(f'Saved frame {frame_count} as {filename}')

            self.root.update_idletasks()
            self.root.update()

        cap.release()

    def start(self):
        video_thread = Thread(target=self.video_loop, daemon=True)
        video_thread.start()
        self.root.mainloop()

# Example usage:
if __name__ == "__main__":
    WasteDetection(capture=0, confidence_threshold=0.7, max_age=30, offset=20, line=320).start()
