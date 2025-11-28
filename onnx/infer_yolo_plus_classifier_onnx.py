# onnx/infer_yolo_plus_classifier_onnx.py
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

CONF_THRESH = 0.25

def load_session(path, providers=None):
    if providers is None:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(path, providers=providers)

def preprocess_det(image_bgr, input_shape):
    h, w = image_bgr.shape[:2]
    size = input_shape[2:]
    img = cv2.resize(image_bgr, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img, (w, h)

def postprocess_det(outputs, orig_size):
    w0, h0 = orig_size
    preds = outputs[0][0]  # (N, 6)
    boxes = []
    for x1, y1, x2, y2, score, cls in preds:
        if score < CONF_THRESH:
            continue
        boxes.append((int(x1), int(y1), int(x2), int(y2), float(score), int(cls)))
    return boxes

def preprocess_cls(crop_bgr):
    img = cv2.resize(crop_bgr, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    # нормализация как в ImageNet
    mean = np.array([0.485, 0.456, 0.406])[None, None, :]
    std = np.array([0.229, 0.224, 0.225])[None, None, :]
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def main():
    det_sess = load_session( "runs/detect/yolov8n_airplanes_base/weights/best.onnx")
    cls_sess = load_session("onnx/aircraft_resnet18_variant.onnx")

    det_input = det_sess.get_inputs()[0]
    dname = det_input.name
    _, _, H, W = det_input.shape

    cname = cls_sess.get_inputs()[0].name

    # список имён классов FGVC Aircraft
    class_names = Path("classifier/fgvc_variant_classes.txt").read_text().splitlines()

    img_path = "test_images/airplane.jpg"
    img = cv2.imread(img_path)
    det_inp, orig_size = preprocess_det(img, (1, 3, H, W))
    det_out = det_sess.run(None, {dname: det_inp})
    boxes = postprocess_det(det_out, orig_size)

    if not boxes:
        print("No airplanes found")
        return

    # возьмём максимум по score
    x1, y1, x2, y2, score, _ = max(boxes, key=lambda b: b[4])
    crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    cls_inp = preprocess_cls(crop)
    logits = cls_sess.run(None, {cname: cls_inp})[0]
    probs = softmax(logits)
    cls_id = int(probs.argmax(axis=1)[0])
    cls_prob = float(probs[0, cls_id])
    cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

    # рисуем рамку + confidence + класс
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{cls_name} {score:.2f}/{cls_prob:.2f}"
    cv2.putText(out, label, (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    Path("results").mkdir(exist_ok=True)
    cv2.imwrite("results/yolo_det_cls_onnx.jpg", out)

if __name__ == "__main__":
    main()
