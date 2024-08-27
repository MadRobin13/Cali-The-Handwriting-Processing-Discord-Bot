from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(data="data.yaml", epochs=3, batch_size=16, img_size=640, device="cpu")

model.export("onnx", "trained_model.onnx")