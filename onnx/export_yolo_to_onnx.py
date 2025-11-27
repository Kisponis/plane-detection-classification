# onnx/export_yolo_to_onnx.py
from ultralytics import YOLO
from pathlib import Path

WEIGHTS = {
    "base": "runs/detect/yolov8n_airplanes_base/weights/best.pt",
    "no_part1": "runs/detect/yolov8n_airplanes_no_part1/weights/best.pt",
    "no_part123": "runs/detect/yolov8n_airplanes_no_part123/weights/best.pt",
}

OUT_DIR = Path("onnx")
OUT_DIR.mkdir(exist_ok=True)

for name, ckpt in WEIGHTS.items():
    model = YOLO(ckpt)
    model.export(
        format="onnx",
        opset=12,
        imgsz=640,
        dynamic=False,
        simplify=True,
        # с включенным постпроцессингом выход обычно (1, N, 6)
    )
    # ultralytics сам кладёт .onnx рядом с ckpt; перенеси в onnx/
