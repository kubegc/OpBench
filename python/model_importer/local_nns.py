import os
import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
# from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor as graph_executor
import python.model_importer.neworkx_visualizer as neworkx_visualizer

os.environ['TVM_BACKTRACE']="1"

class FastSoftmaxMutator(tvm.relay.ExprMutator):
    def __init__(self):
        super().__init__()

    def visit_call(self, call):
        call = super().visit_call(call)
        if isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.softmax":
            return tvm.relay.nn.fast_softmax(call.args[0], call.attrs.axis)
        return call

@tvm.relay.transform.function_pass(opt_level=1)
def FastSoftmax(fn, mod, device):
    return FastSoftmaxMutator().visit(fn)

def get_network(name, batch_size = 1, layout="NCHW", dtype="float32", sequence = 128, hidden_size = 768, num_hidden_layers = 12, num_attention_heads = 12, intermediate_size = 3072, max_position_embeddings = 512):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)
    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    return mod, params, input_shape, output_shape