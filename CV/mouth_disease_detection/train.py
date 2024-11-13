from ultralytics import YOLO

# Load a pre-trained YOLO model (you can choose n, s, m, l, or x versions)
model = YOLO("yolov8n.pt")

# Start training on your custom dataset
model.train(data="/Users/igyuseob/Desktop/capstone/data/구강질환_detection/dataset_all/data/data.yaml", epochs=10, imgsz=640, device="mps")