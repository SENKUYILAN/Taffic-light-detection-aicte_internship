from ultralytics import YOLO
import os
DATA_YAML_PATH = 'Traffic Light Detection.v1i.yolov8/data.yaml'
MODEL_TO_LOAD = 'yolov8n.pt' 
DEVICE = 'cpu' 
EPOCHS = 50
BATCH_SIZE = 16
IMAGE_SIZE = 640
# --- Training Process ---
print(f"Starting training on device: {DEVICE} for {EPOCHS} epochs...")
try:
    # 1. Load a model (starting from scratch with pre-trained weights)
    model = YOLO(MODEL_TO_LOAD)  
    # 2. Start training
    # The results, including the best.pt file, will be saved in runs/detect/train/
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        name='traffic_light_run' # Name your training session
    )

    # 3. Output location confirmation
    output_dir = os.path.join(model.trainer.save_dir, 'weights', 'best.pt')
    print(f"\nTraining complete! Best weights saved to: {output_dir}")
except Exception as e:
    print(f"\n An error occurred during training: {e}")
# Exit if training fails
    BEST_MODEL_PATH = None 
# --- 3. Model Prediction (Runs only if training was successful) ---
if BEST_MODEL_PATH and os.path.exists(BEST_MODEL_PATH):
    print("\n Starting Prediction on Test Images...")
    try:
        trained_model = YOLO(BEST_MODEL_PATH)
        prediction_results = trained_model.predict(
            source=SOURCE_PATH,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            conf=CONFIDENCE_THRESHOLD,
            save=True,  # Saves the annotated images
            name=TRAINING_NAME + '_predict' # Separate folder for prediction results
        )

        print(f"Prediction complete! Results saved to: {trained_model.predictor.save_dir}")

    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")

else:
    print("Skipping prediction because the best model file was not found.")
