import torch
import clip

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224, device=device)

torch.onnx.export(
    model.visual,
    dummy_input,
    "clip_image_encoder.onnx",
    input_names=["image"],
    output_names=["embedding"],
    opset_version=14,
    dynamo=False,
)

print("Exported clip_image_encoder.onnx")
