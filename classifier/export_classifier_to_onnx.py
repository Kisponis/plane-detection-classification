# classifier/export_classifier_to_onnx.py
import torch
from torchvision import models

LEVEL = "variant"
NUM_CLASSES = 100  # подставь фактическое количество (len(train_ds.classes))

ckpt_path = f"classifier/aircraft_resnet18_{LEVEL}_best.pth"
onnx_path = f"onnx/aircraft_resnet18_{LEVEL}.onnx"

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
state = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state)
model.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["images"], output_names=["logits"],
    dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=12,
)
print(f"Saved {onnx_path}")
