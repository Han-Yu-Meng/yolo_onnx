from ultralytics import YOLO
model = YOLO("yolov8n-pose.pt") 
model.export(format="onnx", imgsz=640, opset=12)