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

# python python/performance_collector/op_performance_collector.py --modelsource=local --modelname=resnet-18 --ifcompare=true --tuner=xgb_knob --target=pynq --trials=1000 --host=133.133.135.39 --port=9190 --iftune=true
# python python/performance_collector/op_performance_collector.py --modelsource=local --modelname=resnet-18 --ifcompare=true --tuner=grid
# python python/performance_collector/op_performance_collector.py --modelsource=local --modelname=resnet-18 --ifcompare=true --tuner=xgb --iftune=true --target=cuda --trials=3000
# python python/performance_collector/op_performance_collector.py --modelsource=local --modelname=resnet-18 --ifcompare=true --tuner=xgb --iftune=true --target=pynq --trials=1500 --host=133.133.135.39 --port=9191
# python python/performance_collector/op_performance_collector.py --modelsource=local --modelname=resnet-18 --iftune=true
# python python/performance_collector/op_performance_collector.py --modelsource=local --modelname=resnet-18 --target=cuda
# python python/performance_collector/op_performance_collector.py --modelsource=local --modelname=resnet3d-18 --target=llvm
# python python/performance_collector/op_performance_collector.py --modelsource=local --modelname=resnet3d-18 --target=cuda
# python python/performance_collector/op_performance_collector.py --modelsource=local --modelname=mobilenet --target=llvm
# python python/performance_collector/op_performance_collector.py --modelsource=local --modelname=mobilenet --target=cuda
# python python/performance_collector/op_performance_collector.py --modelsource=local --modelname=squeezenet_v1.1 --target=llvm
# python python/performance_collector/op_performance_collector.py --modelsource=local --modelname=squeezenet_v1.1 --target=cuda
# python python/performance_collector/op_performance_collector.py --modelsource=local --modelname=inception_v3 --target=llvm
# python python/performance_collector/op_performance_collector.py --modelsource=local --modelname=inception_v3 --target=cuda
# python python/performance_collector/op_performance_collector.py --modelsource=transformers --modelname=bert --target=llvm
# python python/performance_collector/op_performance_collector.py --modelsource=transformers --modelname=bert --target=cuda
# python python/performance_collector/op_performance_collector.py --modelsource=transformers --modelname=gpt2 --target=llvm
# python python/performance_collector/op_performance_collector.py --modelsource=transformers --modelname=gpt2 --target=cuda
# python python/performance_collector/op_performance_collector.py --modelsource=transformers --modelname=roberta --target=llvm
# python python/performance_collector/op_performance_collector.py --modelsource=transformers --modelname=roberta --target=cuda
# python python/performance_collector/op_performance_collector.py --modelsource=transformers --modelname=nasnetalarge --target=llvm
# python python/performance_collector/op_performance_collector.py --modelsource=transformers --modelname=nasnetalarge --target=cuda
# python python/performance_collector/op_performance_collector.py --modelsource=transformers --modelname=lstm --target=llvm
# python python/performance_collector/op_performance_collector.py --modelsource=transformers --modelname=lstm --target=cuda
# python python/performance_collector/op_performance_collector.py --modelsource=remoteonnx --modelname=https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/model/t5-decoder-with-lm-head-12.tar.gz --target=llvm
# python python/performance_collector/op_performance_collector.py --modelsource=remoteonnx --modelname=https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/model/t5-decoder-with-lm-head-12.tar.gz --target=cuda
# python python/performance_collector/op_performance_collector.py --modelsource=remoteonnx --modelname=https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/model/t5-encoder-12.tar.gz --target=llvm
# python python/performance_collector/op_performance_collector.py --modelsource=remoteonnx --modelname=https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/model/t5-encoder-12.tar.gz --target=cuda
# python python/performance_collector/op_performance_collector.py --modelsource=simple --modelname=matmul --target=llvm
# python python/performance_collector/op_performance_collector.py --modelsource=simple --modelname=matmul --target=cuda --iftune=true --tuner=grid

def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=10,
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
    autotvm.record.pick_best(tmp_log_file, log_filename)
    # os.remove(tmp_log_file)

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

def run_autoTVM(args,mod):
    if args.target == 'llvm':
        number = 10 #
        repeat = 1 #
        min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
        timeout = 10  # in seconds
        # create a TVM runner
        runner = autotvm.LocalRunner(
            number=number,
            repeat=repeat,
            timeout=timeout,
            min_repeat_ms=min_repeat_ms,
            enable_cpu_cache_flush=True,
            )
        tuning_option = {
            "tuner": args.tuner,
            "trials": args.trials, # 1500,3000
            "early_stopping": None,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(), runner=runner
            ),
            "tuning_records": "/root/github/OpBench/data/Performance/"+args.modelname+ '-' + args.tuner +"-"+str(args.target)+"-autotvm.json",
        }
        tasks = autotvm.task.extract_from_program(mod["main"], target=args.target, params=params)
    elif args.target == 'cuda':
        tuning_option = {
        "log_filename": "/root/github/OpBench/data/Performance/"+args.modelname+ '-' + args.tuner +"-"+str(args.target)+"-autotvm.json",
        "tuner": args.tuner,
        "n_trial": args.trials,
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
            ),
        }
        tasks = autotvm.task.extract_from_program(
        mod["main"], target=args.target, params=params)
        print("Tuning...")
    elif args.target == 'pynq':
        tracker_host = os.environ.get("TVM_TRACKER_HOST", args.host)
        tracker_port = int(os.environ.get("TVM_TRACKER_PORT", args.port))
        # Load VTA parameters from the 3rdparty/vta-hw/config/vta_config.json file
        env = vta.get_env()
        device = "vta"
        target = env.target if device == "vta" else env.target_vta_cpu

        tuning_option = {
            "log_filename": "/root/github/OpBench/data/Performance/"+args.modelname+ '-' + args.tuner +"-"+str(args.target)+"-autotvm.json",
            "tuner": args.tuner,
            "n_trial": args.trials,
            "early_stopping": None,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=100),
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

        tasks = autotvm.task.extract_from_program(
            mod,
            params=params,
            ops=(relay.op.get("nn.conv2d"),),
            target=target,
            target_host=env.target_host,
        )
    findConfigSpace(tasks)
    tune_tasks(tasks, **tuning_option)

def findConfigSpace(tasks):
    fName = "/root/github/OpBench/data/ConfigSpace/"+args.modelname+"-"+str(args.target)+".json"
    f = open(fName, "w")
    for i, task in enumerate(tasks):
        # print(task)
        f.write(task.config_space.__str__())
        # print(task.config_space)
    f.close()
    return

def extra_compile(args, mod, params):
    if args.target == 'pynq':
        with tvm.transform.PassContext(opt_level=3):
            with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
                mod = relay.quantize.quantize(mod, params=params)
        TaskExtractEnv()
        vta_json_env = vta.get_env()
        start_pack = "nn.max_pool2d"
        stop_pack = "nn.global_avg_pool2d"
        # print(mod["main"])
        # print(vta_json_env.TARGET)
        # print(vta_json_env.target)
        assert vta_json_env.BLOCK_IN == vta_json_env.BLOCK_OUT
        relay_prog = graph_pack(
            mod["main"],
            vta_json_env.BATCH,
            vta_json_env.BLOCK_OUT,
            vta_json_env.WGT_WIDTH,
            start_name=start_pack,
            stop_name=stop_pack,
        )
        return relay_prog, tvm.IRModule.from_expr(relay_prog)
    return mod, mod

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

def get_lib_module_dev(args, mod, params):
    target = args.target
    if args.host is None or args.port is None:
        with tvm.transform.PassContext(opt_level=3, config={}):
            lib = relay.build(mod, target=args.target, params=params)
        dev = tvm.device(str(args.target), 0)
        module = graph_executor.GraphModule(lib["default"](dev))
    else:
        env = vta.get_env()
        device = "vta"
        target = env.target if device == "vta" else env.target_vta_cpu
        print(env.target_host)
        print(env.TARGET)
        print(target)
        print("get mod lib")
        if env.TARGET != "sim":
            print("get remote")
            remote = autotvm.measure.request_remote(
            env.TARGET, args.host, args.port, timeout=1000000)
            vta.reconfig_runtime(remote)
            vta.program_fpga(remote, bitstream=None)
        else:
            # In simulation mode, host the RPC server locally.
            remote = rpc.LocalSession()
        if target.device_name != "vta":
            with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
                lib = relay.build(
                    mod,  target=tvm.target.Target(target, host=env.target_host), params=params,
                )
        else:
            print("do here.")
            with vta.build_config(
            opt_level=3, disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}
        ):
                lib = relay.build(
                    relay_prog, target=tvm.target.Target(target, host=env.target_host), params=params, 
                )
        # Export library
        print("Upload...")
        temp = utils.tempdir()
        lib.export_library(temp.relpath("graphlib2.tar"))
        remote.upload(temp.relpath("graphlib2.tar"))
        lib = remote.load_module("graphlib2.tar")
        dev = remote.ext_dev(0) if device == "vta" else remote.cpu(0)
        module = graph_executor.GraphModule(lib["default"](dev))
    return lib, module, target, dev, params

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--modelsource', type=str, default = None)
  parser.add_argument('--modelname', type=str, default=None)
  parser.add_argument('--target', type=str, default="llvm")
  parser.add_argument('--batchsize', type=int, default=1)
  parser.add_argument('--iftune', type=bool, default=False)
  parser.add_argument('--ifcompare', type=bool, default=False)
  parser.add_argument('--tuner', type=str, default='xgb')
  parser.add_argument('--trials', type=int, default=10)
  parser.add_argument('--host', type=str, default=None)
  parser.add_argument('--port', type=int, default=None)
  args = parser.parse_args()
  autotvm.record.encode
  autotvm.measure.MeasureInput
  autotvm.measure.MeasureResult
  if args.target == 'pynq':
    register_vta_tuning_tasks()
    env = vta.get_env()
    device = "vta"
    target = env.target if device == "vta" else env.target_vta_cpu
  if args.modelsource == 'mxnet.vision':
    relay_prog, params = model_importer.local_nns.compile_network(env, target, args.modelname, "nn.max_pool2d", "nn.global_avg_pool2d")
    lib, module, target, dev, params = get_lib_module_dev(args, relay_prog, params)
  if args.modelsource == "local" :
    local_cnns = ["resnet-","resnet3d-","mobilenet","squeezenet_v1.1","inception_v3"]
    if args.modelname.startswith(local_cnns[0]) or args.modelname.startswith(local_cnns[1]) or args.modelname in local_cnns:
        mod, params, input_shape, output_shape = model_importer.local_nns.get_network(args.modelname)
        relay_prog, mod = extra_compile(args, mod, params)
        if args.iftune:
            run_autoTVM(args,mod)
        if args.ifcompare:
            print("compare")
            input_name = "data"
            data = tvm.nd.array((np.random.uniform(size=input_shape)).astype("float32"))
            # lib, module, target, dev, params = get_lib_module_dev(args, relay_prog, params)
            # module.set_input(**params)
            # module.set_input(input_name, data)
            # timeit_performance(module,dev)
            with autotvm.apply_history_best("/root/github/OpBench/data/Performance/"+args.modelname+ '-' + args.tuner +"-"+args.target+"-autotvm.json") as ab:
                print(ab.best_by_model)
                print(ab.best_by_targetkey)
                print(ab._best_user_defined)
                lib, module, target, dev, params = get_lib_module_dev(args, relay_prog, params)
                module.set_input(input_name, data)
                timeit_performance(module,dev)
    else:
        print("error local model name.")
  elif args.modelsource=="transformers":
    # network = "roberta"
    network = args.modelname
    batch_size = args.batchsize
    dtype = "float32"
    target = args.target
    device = tvm.device(str(target), 0)
    if network == 'bert' or network == 'gpt2' or network == 'roberta':
        mod, params, input_shape,inputs = model_importer.transformers_nns.get_network(network, batch_size, dtype=dtype, sequence=128)
        with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
            if args.iftune:
                run_autoTVM(args,mod)
            # lib = relay.build(mod, target=target, params=params)
            # module = graph_executor.create(lib.get_graph_json(),lib.get_lib(), device, dump_root = '/root/github/debug_dump/' + network)
            # input_ids = tvm.nd.array((np.random.uniform(size=input_shape)).astype("int64"))
            # module.set_input("input_ids", input_ids)
            # print("Evaluate inference time cost...")
            # module.run()
    elif network == "nasnetalarge":
            #target = tvm.target.Target("llvm -mcpu=core-avx2")
            # target = tvm.target.Target("llvm -mcpu=skylake-avx512")
        mod, params, input_shape,inputs = model_importer.transformers_nns.get_network(network, batch_size, dtype=dtype, sequence=128)
        with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
            if args.iftune:
                run_autoTVM(args,mod)
            # lib = relay.build(mod, target=target, params=params)
            # module = graph_executor.create(lib.get_graph_json(),lib.get_lib(), device, dump_root = '/root/github/debug_dump/' + network)
            # input_ids = tvm.nd.array((np.random.uniform(size=input_shape)).astype("float32"))
            # module.set_input("input0", input_ids)
            # attention_mask = tvm.nd.array((np.random.uniform(size=shape2)).astype("int64"))
            # module.set_input("attention_mask", attention_mask)
            # module.set_input("decoder_input_ids", input_ids)
            # print("Evaluate inference time cost...")
            # module.run()
    elif network == 'lstm' or network == 'rnn' or network == 'gru':
        mod, params, input_shape,inputs = model_importer.transformers_nns.get_network(network, batch_size, dtype=dtype, sequence=128)
        with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
            if args.iftune:
                run_autoTVM(args,mod)
            # lib = relay.build(mod, target=target, params=params)
            # module = graph_executor.create(lib.get_graph_json(),lib.get_lib(), device, dump_root = '/root/github/debug_dump/' + network)
            # for key in inputs:
            #     module.set_input(key, tvm.nd.array(inputs[key].astype("float32")))
            # print("Evaluate inference time cost...")
            # module.run()
    elif network == 'dpn68':
        mod, params, input_shape,inputs = model_importer.transformers_nns.get_network(network, batch_size, dtype=dtype, sequence=128)
        with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": False}):
            if args.iftune:
                run_autoTVM(args,mod)
            # lib = relay.build(mod, target=target, params=params)
            # # module = graph_executor.GraphModule(lib["default"](device))
            # module = graph_executor.create(lib.get_graph_json(),lib.get_lib(), device, dump_root = '/root/github/debug_dump/' + network)
            # input_ids = tvm.nd.array((np.random.uniform(size=input_shape)).astype("float32"))
            # module.set_input("input0", input_ids)
            # print("Evaluate inference time cost...")
            # module.run()
    else:
        print("error transformers models.")
  elif args.modelsource=="remoteonnx":
      netework, mod, params, lib, module = model_importer.onnx_nns.get_onnx_with_url(args.modelname, args.target, batch = args.batchsize, sequence = 128,  if_run = False)
      args.modelname = netework
      if args.iftune:
        run_autoTVM(args,mod)
  elif args.modelsource=="simple":
    mod, params, input_names, inputs = model_importer.simple_nns.get_simple_network(args.modelname,args.target)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=args.target, params=params)
    dev = tvm.device(str(args.target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    for i in range(len(input_names)):
        module.set_input(input_names[i], inputs[i])
    if args.iftune:
        run_autoTVM(args,mod)
    if args.ifcompare:
        # timeit_performance(module)
        full_log = "/root/github/OpBench/data/Performance/"+args.modelname+ '-' + args.tuner +"-"+args.target+"-autotvm.json"
        single_log = full_log+".tmp"
        print(full_log)
        autotvm.record.pick_best(full_log, single_log)
        # with autotvm.apply_history_best(single_log):
        #   with tvm.transform.PassContext(opt_level=3, config={}):
        #     lib = relay.build(mod, target=args.target, params=params)
        # dev = tvm.device(str(args.target), 0)
        # module = graph_executor.GraphModule(lib["default"](dev))
        # for i in range(len(input_names)):
        #     module.set_input(input_names[i], inputs[i])
        # timeit_performance(module)
        # timeit_performance(module)
        # timeit_performance(module)
  else:
      print("error model source.")

# base_path = '/root/github/onnx-models/'
# dump_path = '/root/github/debug_dump/'
# target = "llvm"

# model_url = (
#     "https://github.com/onnx/models/raw/main/"
#     "vision/classification/resnet/model/"
#     "resnet50-v2-7.onnx"
# )

# model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
# onnx_model = onnx.load(model_path)

# Seed numpy's RNG to get consistent results
np.random.seed(0)

# img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
# img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

# Resize it to 224x224
# resized_image = Image.open(img_path).resize((224, 224))
# img_data = np.asarray(resized_image).astype("float32")

# Our input image is in HWC layout while ONNX expects CHW input, so convert the array
# img_data = np.transpose(img_data, (2, 0, 1))

# Normalize according to the ImageNet input specification
# imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
# imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
# norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

# Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
# img_data = np.expand_dims(norm_img_data, axis=0)

# The input name may vary across model types. You can use a tool
# like Netron to check input names

# mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# dtype = "float32"
# module.set_input(input_name, img_data)
# module.run()
# output_shape = (1, 1000)
# tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()