import onnx
import transformers
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
import argparse
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm import autotvm
import os
import model_importer.local_nns 
import model_importer.transformers_nns
import model_importer.onnx_nns
import model_importer.simple_nns

# python python/performance_collector/config_space_exporter.py --modelsource=local --modelname=resnet-18 --iftune=true
# python python/performance_collector/config_space_exporter.py --modelsource=local --modelname=resnet-18 --target=cuda
# python python/performance_collector/config_space_exporter.py --modelsource=local --modelname=resnet3d-18 --target=llvm
# python python/performance_collector/config_space_exporter.py --modelsource=local --modelname=resnet3d-18 --target=cuda
# python python/performance_collector/config_space_exporter.py --modelsource=local --modelname=mobilenet --target=llvm
# python python/performance_collector/config_space_exporter.py --modelsource=local --modelname=mobilenet --target=cuda
# python python/performance_collector/config_space_exporter.py --modelsource=local --modelname=squeezenet_v1.1 --target=llvm
# python python/performance_collector/config_space_exporter.py --modelsource=local --modelname=squeezenet_v1.1 --target=cuda
# python python/performance_collector/config_space_exporter.py --modelsource=local --modelname=inception_v3 --target=llvm
# python python/performance_collector/config_space_exporter.py --modelsource=local --modelname=inception_v3 --target=cuda
# python python/performance_collector/config_space_exporter.py --modelsource=transformers --modelname=bert --target=llvm
# python python/performance_collector/config_space_exporter.py --modelsource=transformers --modelname=bert --target=cuda
# python python/performance_collector/config_space_exporter.py --modelsource=transformers --modelname=gpt2 --target=llvm
# python python/performance_collector/config_space_exporter.py --modelsource=transformers --modelname=gpt2 --target=cuda
# python python/performance_collector/config_space_exporter.py --modelsource=transformers --modelname=roberta --target=llvm
# python python/performance_collector/config_space_exporter.py --modelsource=transformers --modelname=roberta --target=cuda
# python python/performance_collector/config_space_exporter.py --modelsource=transformers --modelname=nasnetalarge --target=llvm
# python python/performance_collector/config_space_exporter.py --modelsource=transformers --modelname=nasnetalarge --target=cuda
# python python/performance_collector/config_space_exporter.py --modelsource=transformers --modelname=lstm --target=llvm
# python python/performance_collector/config_space_exporter.py --modelsource=transformers --modelname=lstm --target=cuda
# python python/performance_collector/config_space_exporter.py --modelsource=remoteonnx --modelname=https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/model/t5-decoder-with-lm-head-12.tar.gz --target=llvm
# python python/performance_collector/config_space_exporter.py --modelsource=remoteonnx --modelname=https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/model/t5-decoder-with-lm-head-12.tar.gz --target=cuda
# python python/performance_collector/config_space_exporter.py --modelsource=remoteonnx --modelname=https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/model/t5-encoder-12.tar.gz --target=llvm
# python python/performance_collector/config_space_exporter.py --modelsource=remoteonnx --modelname=https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/model/t5-encoder-12.tar.gz --target=cuda
# python python/performance_collector/config_space_exporter.py --modelsource=simple --modelname=matmul --target=llvm
# python python/performance_collector/config_space_exporter.py --modelsource=simple --modelname=matmul --target=cuda

def run_autoTVM(args,mod):
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
        "tuner": "gridsearch",
        "trials": 3000, # 1500,3000
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
        "tuning_records": "/root/github/OpBench/data/Performance/"+args.modelname+"-"+args.target+"-autotvm.json",
    }
    tasks = autotvm.task.extract_from_program(mod["main"], target=args.target, params=params)
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner_obj = GridSearchTuner(task)
        tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
            ],
        )
    return

def findConfigSpace(args,mod):
    tasks = autotvm.task.extract_from_program(mod["main"], target=args.target, params=params)
    fName = "/root/github/OpBench/data/ConfigSpace/"+args.modelname+"-"+args.target+".json"
    f = open(fName, "w")
    for i, task in enumerate(tasks):
        print(task)
        f.write(task.config_space.__str__())
        # print(task.config_space)
    f.close();
    return

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--modelsource', type=str, default = None)
  parser.add_argument('--modelname', type=str, default=None)
  parser.add_argument('--target', type=str, default="llvm")
  parser.add_argument('--batchsize', type=int, default=1)
  parser.add_argument('--iftune', type=bool, default=False)
  args = parser.parse_args()
  if args.modelsource=="local":
    local_cnns = ["resnet-","resnet3d-","mobilenet","squeezenet_v1.1","inception_v3"]
    if args.modelname.startswith(local_cnns[0]) or args.modelname.startswith(local_cnns[1]) or args.modelname in local_cnns:
        mod, params, input_shape, output_shape = model_importer.local_nns.get_network(args.modelname)
        input_name = "data"
        shape_dict = {input_name: input_shape}
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=args.target, params=params)
            dev = tvm.device(str(args.target), 0)
            module = graph_executor.GraphModule(lib["default"](dev))
            findConfigSpace(args,mod)
            if args.iftune:
                run_autoTVM(args,mod)
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
            findConfigSpace(args, mod)
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
            findConfigSpace(args, mod)
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
            findConfigSpace(args, mod)
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
            findConfigSpace(args, mod)
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
      findConfigSpace(args, mod)
      if args.iftune:
        run_autoTVM(args,mod)
  elif args.modelsource=="simple":
      mod, lib, module, params = model_importer.simple_nns.get_simple_network(args.modelname,args.target)
      findConfigSpace(args, mod)
      run_autoTVM(args,mod)
      if args.iftune:
        run_autoTVM(args,mod)
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

# import timeit

# timing_number = 10
# timing_repeat = 10
# unoptimized = (
#     np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
#     * 1000
#     / timing_number
# )
# unoptimized = {
#     "mean": np.mean(unoptimized),
#     "median": np.median(unoptimized),
#     "std": np.std(unoptimized),
# }

# print(unoptimized)
    # tuner_obj.tune(
    #     n_trial=min(tuning_option["trials"], len(task.config_space)),
    #     early_stopping=tuning_option["early_stopping"],
    #     measure_option=tuning_option["measure_option"],
    #     callbacks=[
    #         autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
    #         autotvm.callback.log_to_file(tuning_option["tuning_records"]),
    #     ],
    # )