import os
import onnx
import onnx.helper
import onnx.printer


value = [2, 15, 13]
const_tensor = onnx.helper.make_tensor(
    name='const_tensor',
    data_type= onnx.TensorProto.INT64,
    dims=[3],  # 提供张量的维度
    vals=value  # 实际的数值
)

# 使用helper函数创建Constant Node
const_node =  onnx.helper.make_node(
    'Constant',  # ONNX操作类型，对于常量节点固定为'Constant'
    [],          # 输入节点列表，常量节点没有输入
    ['V1'],  # 输出节点名称列表，这里假设输出名为'output'
    name='const_node',  # 节点名称
    value=const_tensor  # 引入上面创建的TensorProto对象
)

graph_proto = onnx.helper.make_graph(
    [
        const_node,
        onnx.helper.make_node("SplitV13", ["V0", "V1"], ["V2", "V3", "V4"], axis=1, num_outputs=3),
        #onnx.helper.make_node("Sub", ["V1", "V2"], ["V3"]),
        #onnx.helper.make_node("Concat", ["V0", "V1", "V2"], ["V4"], axis=1),
        #onnx.helper.make_node("Div", ["R2", "R2"], ["R3"]),
        #onnx.helper.make_node("Add", ["A0", "A1"], ["R4"]),
        #onnx.helper.make_node("Add", ["R3", "R4"], ["R5"]),
    ],
    "split",
    [
        onnx.helper.make_tensor_value_info("V0" , onnx.TensorProto.FLOAT16, [1, 30, 2, 2]),
        #onnx.helper.make_tensor_value_info("V1" , onnx.TensorProto.INT64, [3]),
    ],
    [
        onnx.helper.make_tensor_value_info("V2" , onnx.TensorProto.FLOAT16, [1, 2, 2, 2]),
        onnx.helper.make_tensor_value_info("V3" , onnx.TensorProto.FLOAT16, [1, 15, 2, 2]),
        onnx.helper.make_tensor_value_info("V4" , onnx.TensorProto.FLOAT16, [1, 13, 2, 2]),
    ],
    None,
    None,
    [
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