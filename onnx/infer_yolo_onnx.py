# onnx/infer_yolo_onnx.py
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

CONF_THRESH = 0.25

def load_session(onnx_path: str) -> ort.InferenceSession:
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],  # или CUDAExecutionProvider
    )
    return sess

def preprocess(image_bgr, input_shape):
    h, w = image_bgr.shape[:2]
    size = input_shape[2:]  # (H, W)
    img = cv2.resize(image_bgr, size, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)   # NCHW
    return img, (w, h)

def postprocess(outputs, orig_size):
    w0, h0 = orig_size
    preds = outputs[0]  # (1, N, 6)
    preds = preds[0]
    boxes = []
    for x1, y1, x2, y2, score, cls in preds:
        if score < CONF_THRESH:
            continue
        # координаты уже в пикселях исходного размера (при экспорте с постпроцессингом)
        boxes.append((int(x1), int(y1), int(x2), int(y2), float(score), int(cls)))
    return boxes

def draw_boxes(image_bgr, boxes, class_name="plane"):
    for x1, y1, x2, y2, score, cls in boxes:
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {score:.2f}"
        cv2.putText(image_bgr, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image_bgr

def main():
    onnx_path = "onnx/yolov8n_airplanes_base.onnx"
    img_path = "test_images/airplane.jpg"

    sess = load_session(onnx_path)
    input_name = sess.get_inputs()[0].name
    _, _, H, W = sess.get_inputs()[0].shape

    img_bgr = cv2.imread(img_path)
    inp, orig_size = preprocess(img_bgr, (1, 3, H, W))
    outputs = sess.run(None, {input_name: inp})
    boxes = postprocess(outputs, orig_size)
    out = draw_boxes(img_bgr.copy(), boxes)

    Path("results").mkdir(exist_ok=True)
    cv2.imwrite("results/yolo_onnx_detection.jpg", out)

if __name__ == "__main__":
    main()
