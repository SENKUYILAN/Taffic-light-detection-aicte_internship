from ultralytics import YOLO

# Loading the pre-trained model
model = YOLO('yolov8n.pt')

#Define the training parameters and start training
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='traffic_light'
)

