from ultralytics import YOLO
import torch

def train(model_cfg: str, run_name: str):
    # Проверим, виден ли MPS
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"  # запасной вариант

    print(f"Using device: {device}")

    model = YOLO(model_cfg)
    model.train(
        data="yolo/coco_airplanes.yaml",
        epochs=50,
        imgsz=640,
        batch=32,
        name=run_name,
        lr0=0.01,
        cos_lr=True,
        workers=4,          # я бы уменьшил на маке
        device=device,      # ВАЖНО
    )

if __name__ == "__main__":
    train("yolov8n.pt", run_name="yolov8n_airplanes_base")
