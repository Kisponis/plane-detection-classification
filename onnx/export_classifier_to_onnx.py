from ultralytics import YOLO
from pathlib import Path

WEIGHTS = {
    "base": "runs/detect/yolov8n_airplanes_base/weights/best.pt",
    "no_part1": "runs/detect/yolov8n_airplanes_no_part1/weights/best.pt",
    "no_part123": "runs/detect/yolov8n_airplanes_no_part123/weights/best.pt",
}

Path("onnx").mkdir(exist_ok=True)

for name, ckpt in WEIGHTS.items():
    model = YOLO(ckpt)
    model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        name=f"onnx/yolov8n_airplanes_{name}"
    )

