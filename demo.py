from ultralytics import YOLO

model = YOLO("best12cls.pt")

model.export(format="engine")