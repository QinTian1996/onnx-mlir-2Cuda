import os
import onnx
import onnx.helper
import onnx.printer

graph_proto = onnx.helper.make_graph(
    [
        onnx.helper.make_node("Add", ["A0", "A1"], ["R0"]),
        onnx.helper.make_node("Sub", ["A1", "R0"], ["R1"]),
        onnx.helper.make_node("Mul", ["R0", "R1"], ["R2"]),
        onnx.helper.make_node("Div", ["R1", "R2"], ["R3"]),
        onnx.helper.make_node("Add", ["A0", "A1"], ["R4"]),
        onnx.helper.make_node("Add", ["R3", "R4"], ["R5"]),
    ],
    "ADD",
    [
        onnx.helper.make_tensor_value_info("A0" , onnx.TensorProto.FLOAT, [5, 6, 7]),
        onnx.helper.make_tensor_value_info("A1" , onnx.TensorProto.FLOAT, [5, 6, 7]),
    ],
    [
        onnx.helper.make_tensor_value_info("R5", onnx.TensorProto.FLOAT, [5, 6, 7]),
    ],
    None,
    None,
    [
        onnx.helper.make_tensor_value_info("R0" , onnx.TensorProto.FLOAT, [5, 6, 7]),
        onnx.helper.make_tensor_value_info("R1" , onnx.TensorProto.FLOAT, [5, 6, 7]),
        onnx.helper.make_tensor_value_info("R2" , onnx.TensorProto.FLOAT, [5, 6, 7]),
        onnx.helper.make_tensor_value_info("R3" , onnx.TensorProto.FLOAT, [5, 6, 7]),
        onnx.helper.make_tensor_value_info("R4" , onnx.TensorProto.FLOAT, [5, 6, 7]),
    ],
)

#print("\ngraph proto:\n")
#print(graph_proto)

#print("\nMore Readable GraphProto:\n")
#print(helper.printable_graph(graph_proto))

model = onnx.helper.make_model(graph_proto)
print(graph_proto.name)
onnx.save_model(model, graph_proto.name + ".onnx")
print(onnx.printer.to_text(model))