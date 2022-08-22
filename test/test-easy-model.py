# test easy-model
import tvm
from tvm import relay
import onnx
import os
from tvm.contrib import graph_executor

name="easy-model-fish"
input_info={
    "name": "input_edge",
    "shape": (4,3,14,14)
}

easy_model=onnx.load("/root/github/OpBench/test/easy-model/easy-model-fish.onnx")
mod, params = relay.frontend.from_onnx(easy_model,{"input_edge": (4,3,14,14)})
irModule=relay.transform.InferType()(mod)                    # tvm.ir.module.IRModule

with tvm.transform.PassContext(opt_level=0):
    intrp = relay.build_module.create_executor("graph", irModule, tvm.cpu(0), target=drivers.CPU.target,params=params)


with tvm.transform.PassContext(opt_level=0):
    lib = relay.build(irModule, target="llvm", params={})
