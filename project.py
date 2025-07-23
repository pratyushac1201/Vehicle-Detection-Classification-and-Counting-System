from ultralytics import YOLO


# Load a model
model = YOLO("yolov8s.pt")  # build a new model from scratch
# Use the model
model.train(data="config.yaml", epochs=10)  # train the model




