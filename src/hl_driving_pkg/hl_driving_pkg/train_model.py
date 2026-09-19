from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.train(
    data="/home/yeong/Downloads/2026_TrafficLight.v2i.yolov11/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0
)