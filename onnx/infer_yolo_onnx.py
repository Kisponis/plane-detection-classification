# onnx/infer_yolo_onnx.py
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

CONF_THRESH = 0.25

def load_session(onnx_path: str) -> ort.InferenceSession:
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],  # у тебя и так всё на CPU
    )
    return sess

def preprocess(image_bgr, input_shape):
    h, w = image_bgr.shape[:2]
    _, _, H, W = input_shape  # (1, 3, H, W)
    img = cv2.resize(image_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)   # NCHW
    return img, (w, h), (W, H)

def postprocess(outputs, orig_size, input_size):
    """
    outputs[0]: (1, 5, 8400) -> (8400, 5)
    каждая строка: [cx, cy, w, h, score]
    """
    w0, h0 = orig_size
    W, H = input_size  # размер, в который ресайзили (обычно 640x640)

    preds = outputs[0]      # (1, 5, 8400)
    preds = preds[0]        # (5, 8400)
    preds = preds.transpose(1, 0)  # (8400, 5)

    boxes = []
    for cx, cy, bw, bh, score in preds:
        if score < CONF_THRESH:
            continue

        # координаты в пространстве входа модели (0..W, 0..H)
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # масштабируем к оригинальному размеру картинки
        x1 = x1 * (w0 / W)
        x2 = x2 * (w0 / W)
        y1 = y1 * (h0 / H)
        y2 = y2 * (h0 / H)

        # к клипу, на всякий
        x1 = max(0, min(w0 - 1, x1))
        x2 = max(0, min(w0 - 1, x2))
        y1 = max(0, min(h0 - 1, y1))
        y2 = max(0, min(h0 - 1, y2))

        boxes.append((int(x1), int(y1), int(x2), int(y2), float(score), 0))  # cls=0

    return boxes

def draw_boxes(image_bgr, boxes, class_name="plane"):
    for x1, y1, x2, y2, score, cls in boxes:
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {score:.2f}"
        cv2.putText(
            image_bgr, label, (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    return image_bgr

def main():
    onnx_path = "runs/detect/yolov8n_airplanes_base/weights/best.onnx"
    img_path = "data/coco_airplanes/images/val/000000001761.jpg"

    sess = load_session(onnx_path)
    input_tensor = sess.get_inputs()[0]
    input_name = input_tensor.name
    input_shape = input_tensor.shape  # (1, 3, H, W)

    img_bgr = cv2.imread(img_path)
    inp, orig_size, input_hw = preprocess(img_bgr, input_shape)
    outputs = sess.run(None, {input_name: inp})
    boxes = postprocess(outputs, orig_size, input_hw)
    out = draw_boxes(img_bgr.copy(), boxes)

    Path("results").mkdir(exist_ok=True)
    cv2.imwrite("results/yolo_onnx_detection.jpg", out)
    print(f"Saved results/yolo_onnx_detection.jpg with {len(boxes)} boxes")

if __name__ == "__main__":
    main()

