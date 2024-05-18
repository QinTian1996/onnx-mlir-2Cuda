import onnx

print(onnx.load_model("../mobilenet_v1_1.0_224.onnx").graph.node)
