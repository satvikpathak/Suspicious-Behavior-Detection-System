import onnx

# Load and check the model
model_path = "emotion_model.onnx"
try:
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("ONNX model is valid.")
    print(f"ONNX opset version: {model.opset_import[0].version}")
except Exception as e:
    print(f"ONNX model validation failed: {e}")