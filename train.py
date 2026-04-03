from ultralytics import YOLO

def train_model():
    model = YOLO("yolo11n-seg.pt") 

    # Use the 'r' prefix for the string to handle Windows backslashes
    yaml_path = r"E:\Smoke-and-FireClassifier-main\datasets\data\data.yaml"

    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0
    )

if __name__ == "__main__":
    train_model()