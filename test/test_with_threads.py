import onnx
import transformers
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor, utils, download
import argparse
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm import rpc, autotvm, relay, te, topi
import os
import model_importer.local_nns 
import model_importer.transformers_nns
import model_importer.onnx_nns
import model_importer.simple_nns
import relay_profiler.util
import vta
from vta.testing import simulator
from vta.top import graph_pack
from tvm.autotvm.task import TaskExtractEnv

mod, params, input_shape, output_shape = model_importer.local_nns.get_network("resnet-18")

def timeit_performance(module, ctx):
    import timeit
    timing_number = 10
    timing_repeat = 10
    print("ready to run")
    timer = module.module.time_evaluator("run", ctx, number=timing_number, repeat=timing_repeat)
    unoptimized = np.array(timer().results) * 1000 / timing_number
    print("runned")
    unoptimized = {
        "mean": np.mean(unoptimized),
        "median": np.median(unoptimized),
        "std": np.std(unoptimized),
    }
    print(unoptimized)
    return unoptimized

# tvm.target.cuda() 
# tvm.target.Target
cuda = "cuda -keys=cuda,gpu -max_num_threads=1024 -thread_warp_size=1024"
llvm = "llvm -keys=cpu -link-params=0"

with tvm.transform.PassContext(opt_level=3, config={}):
    lib = relay.build(mod, cuda, params=params)
    dev = tvm.device(str(cuda), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    data = tvm.nd.array((np.random.uniform(size=input_shape)).astype("float32"))
    module.set_input("data", data)
    timeit_performance(module,dev)

