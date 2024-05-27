import onnx
import sys
import io
import re

def Usage():
    print("Usage: python ", sys.argv[0], " model_filename")
    exit(1)


if len(sys.argv) < 2:
    Usage()

path = sys.argv[1]
text = io.StringIO()
print(onnx.load_model(path).graph.node, file=text)
v=text.getvalue()
ops = set()
for line in v.splitlines():
    if re.search("op_type", line):
        ops.add(line[10:-1])
print(sorted(ops))
for i in sorted(ops):
    print(i)