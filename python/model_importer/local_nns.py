import os
from turtle import window_height
import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
# from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor as graph_executor
import model_importer.neworkx_visualizer as neworkx_visualizer
from vta.top import graph_pack
# from mxnet.gluon.model_zoo import vision
import numpy as np
from yolort.models import yolov5n
from yolort.relay import get_trace_module

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

def compile_network(env, target, model, start_pack, stop_pack):
    
    pack_dict = {
    "resnet18_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet18_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet50_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet101_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    }
    assert model in pack_dict
    # Populate the shape and data type dictionary
    dtype_dict = {"data": "float32"}
    shape_dict = {"data": (env.BATCH, 3, 224, 224)}

    # Get off the shelf gluon model, and convert to relay
    gluon_model = vision.get_model(model, pretrained=True)
    mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

    # Update shape and type dictionary
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    # Perform quantization in Relay
    # Note: We set opt_level to 3 in order to fold batch norm
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params=params)

    # Perform graph packing and constant folding for VTA target
    if target.device_name == "vta":
        assert env.BLOCK_IN == env.BLOCK_OUT
        relay_prog = graph_pack(
            mod["main"],
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name=start_pack,
            stop_name=stop_pack,
        )
        return relay_prog, params
    return mod["main"], params

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
    elif name == "yolov5n":
        in_size = 640
        input_shape = (in_size, in_size)
        model_func = yolov5n(pretrained=True, size=(in_size, in_size))
        script_module = get_trace_module(model_func, input_shape=input_shape)
        input_name = "input0"
        shape_list = [(input_name, (1, 3, *input_shape))]
        mod, params = relay.frontend.from_pytorch(script_module, shape_list)
        output_shape = ""
    elif name == "yolov3":
        from tvm.contrib.download import download_testdata
        from tvm.relay.testing.darknet import __darknetffi__
        MODEL_NAME = "yolov3-tiny"
        REPO_URL = "https://github.com/dmlc/web-data/blob/main/darknet/"

        cfg_path = download_testdata(
            "https://github.com/pjreddie/darknet/blob/master/cfg/" + MODEL_NAME + ".cfg" + "?raw=true",
            MODEL_NAME + ".cfg",
            module="darknet",
        )
        # cfg_path ="/root/.tvm_test_data/darknet/yolov3-tiny.cfg"
        weights_path = download_testdata(
            "https://pjreddie.com/media/files/" + MODEL_NAME + ".weights" + "?raw=true",
            MODEL_NAME + ".weights",
            module="darknet",
        )
        # weights_path = "/root/.tvm_test_data/darknet/yolov3-tiny.weights"

        darknet_lib_path = download_testdata(
            REPO_URL + "lib/" + "libdarknet2.0.so" + "?raw=true", "libdarknet2.0.so", module="darknet"
        )

        net = __darknetffi__.dlopen(darknet_lib_path).load_network(
        cfg_path.encode("utf-8"), weights_path.encode("utf-8"), 0
        )
        # dshape = (env.BATCH, net.c, net.h, net.w)
        # (1,3,416,416)
        input_shape = (1, net.c, net.h, net.w)
        print(input_shape)
        dtype = "float32"
        #(1,1000)
        output_shape=""
        # Start front end compilation
        mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=input_shape)
    return mod, params, input_shape, output_shape