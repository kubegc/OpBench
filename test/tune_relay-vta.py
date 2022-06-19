import os
from mxnet.gluon.model_zoo import vision
import numpy as np
from PIL import Image

from tvm import topi
import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_executor, utils, download
from tvm.autotvm.measure.measure_methods import request_remote
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

import vta
from vta.testing import simulator
from vta.top import graph_pack

#################################################################
# Compile network
# ---------------
# Perform vta-specific compilation with Relay from a Gluon model


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

# Tracker host and port can be set by your environment
tracker_host = "133.133.135.39"
tracker_port =  9190

# Load VTA parameters from the 3rdparty/vta-hw/config/vta_config.json file
env = vta.get_env()

# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

# Name of Gluon model to compile
# The ``start_pack`` and ``stop_pack`` labels indicate where
# to start and end the graph packing relay pass: in other words
# where to start and finish offloading to VTA.
network = "resnet18_v1"
start_pack = "nn.max_pool2d"
stop_pack = "nn.global_avg_pool2d"

# Tuning option
log_file = "%s.%s.log" % (device, network)
print(env.TARGET)
tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb_knob",
    "n_trial": 10,
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.RPCRunner(
            env.TARGET,
            host=tracker_host,
            port=tracker_port,
            number=5,
            timeout=60,
            module_loader=vta.module_loader(),
            # check_correctness=True, # TODO: re-enable when check_correctness works again.
        ),
    ),
}

def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


########################################################################
# Register VTA-specific tuning tasks


def register_vta_tuning_tasks():
    from tvm.autotvm.task import TaskExtractEnv

    @tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
    def my_clip(x, a_min, a_max):
        """Unlike topi's current clip, put min and max into two stages."""
        const_min = tvm.tir.const(a_min, x.dtype)
        const_max = tvm.tir.const(a_max, x.dtype)
        x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
        x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
        return x

    # init autotvm env to register VTA operator
    TaskExtractEnv()

    @autotvm.template("conv2d_packed.vta")
    def _topi_nn_conv2d(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        A, W = args[:2]

        with tvm.target.vta():
            res = vta.top.conv2d_packed(*args, **kwargs)
            res = topi.right_shift(res, 8)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == "vta":
            s = vta.top.schedule_conv2d_packed([res])
        else:
            s = te.create_schedule([res.op])
        return s, [A, W, res]


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.


def tune_and_evaluate(tuning_opt):

    # Register VTA tuning tasks
    register_vta_tuning_tasks()

    # Perform task extraction on Relay program
    print("Extract tasks...")
    relay_prog, params = compile_network(env, target, network, start_pack, stop_pack)
    mod = tvm.IRModule.from_expr(relay_prog)
    tasks = autotvm.task.extract_from_program(
        mod,
        params=params,
        ops=(relay.op.get("nn.conv2d"),),
        target=target,
        target_host=env.target_host,
    )

    # filter out non-packed conv2d task
    tasks = list(filter(lambda t: len(t.args[0][1]) > 4 and "conv" in t.name, tasks))

    # We should have extracted 10 convolution tasks
    assert len(tasks) == 10
    print("Extracted {} conv2d tasks:".format(len(tasks)))
    for tsk in tasks:
        inp = tsk.args[0][1]
        wgt = tsk.args[1][1]
        batch = inp[0] * inp[4]
        in_filter = inp[1] * inp[5]
        out_filter = wgt[0] * wgt[4]
        height, width = inp[2], inp[3]
        hkernel, wkernel = wgt[2], wgt[3]
        hstride, wstride = tsk.args[2][0], tsk.args[2][1]
        hpad, wpad = tsk.args[3][0], tsk.args[3][1]
        print(
            "({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
                batch,
                height,
                width,
                in_filter,
                out_filter,
                hkernel,
                wkernel,
                hpad,
                wpad,
                hstride,
                wstride,
            )
        )

    # We do not run the tuning in our webpage server since it takes too long.
    # Comment the following line to run it by yourself.
    # return

    # run tuning tasks
    # print("Tuning...")
    # tune_tasks(tasks, **tuning_opt)

    # evaluate with tuning history
    if env.TARGET != "sim":
        # Get remote from fleet node
        remote = autotvm.measure.request_remote(
            env.TARGET, tracker_host, tracker_port, timeout=1000000
        )
        # Reconfigure the JIT runtime and FPGA.
        vta.reconfig_runtime(remote)
        vta.program_fpga(remote, bitstream=None)
    else:
        # In simulation mode, host the RPC server locally.
        remote = rpc.LocalSession()

    # compile kernels with history best records
    # with autotvm.tophub.context(target, extra_files=[log_file]):
        # Compile network
        print("Compile...")
    if target.device_name != "vta":
        with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
            lib = relay.build(
                    relay_prog,  target=tvm.target.Target(target, host=env.target_host), params=params,
                )
    else:
        with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
            lib = relay.build(
                    relay_prog, target=tvm.target.Target(target, host=env.target_host), params=params, 
                )

        # Export library
        print("Upload...")
        temp = utils.tempdir()
        lib.export_library(temp.relpath("graphlib.tar"))
        remote.upload(temp.relpath("graphlib.tar"))
        lib = remote.load_module("graphlib.tar")

        # Generate the graph executor
        ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)
        m = graph_executor.GraphModule(lib["default"](ctx))

        # upload parameters to device
        image = tvm.nd.array((np.random.uniform(size=(1, 3, 224, 224))).astype("float32"))
        m.set_input("data", image)

        # evaluate
        print("Evaluate inference time cost...")
        timer = m.module.time_evaluator("run", ctx, number=1, repeat=10)
        tcost = timer()
        prof_res = np.array(tcost.results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )


# Run the tuning and evaluate the results
tune_and_evaluate(tuning_option)
