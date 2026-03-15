from ultralytics import YOLO

model = YOLO("models/yolov8_hand_pose.pt")

model.export(format="onnx")