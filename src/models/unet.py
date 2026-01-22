import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(
    "mobilenetv2_unet_jetson.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Input dummy (igual que en export)
x = np.random.randn(1, 5, 256, 256).astype(np.float32)

outputs = sess.run(None, {"input": x})
y = outputs[0]

print(y.shape)  # (1, 2, 256, 256)
