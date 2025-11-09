from ultralytics import YOLO

# 1. Load the trained model weights
MODEL_PATH = 'runs/detect/traffic_light_final_run/weights/best.pt' 
model = YOLO(MODEL_PATH)

# 2. Define prediction source and parameters
# 'test/images' tells it to process all images in that folder
SOURCE_PATH = 'test/images'
IMAGE_SIZE = 640
DEVICE = 'cpu'

# 3. Run the prediction
results = model.predict(
    source=SOURCE_PATH,
    imgsz=IMAGE_SIZE,
    device=DEVICE,
    save=True,     
    show=False,     
    conf=0.25       
)

print(f"Prediction complete! Results saved to: {model.predictor.save_dir}")
