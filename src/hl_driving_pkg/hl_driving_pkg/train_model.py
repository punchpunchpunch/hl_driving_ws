from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.train(
    data="/home/yeong/DynamicMotors_TrafficLight.v6i.yolov11/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)