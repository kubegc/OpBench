from html import entities
import pandas as pd
import numpy as np
import json
from tvm.autotvm.record import decode
from tvm.autotvm import task
import argparse

# 获取配置中的threads和tile size
# log format
# costs 和 all_cost区别
    # if protocol == "json":
    #     json_dict = {
    #         "input": (str(inp.target), inp.task.name, inp.task.args, inp.task.kwargs),
    #         "config": inp.config.to_json_dict(),
    #         "result": (
    #             result.costs if result.error_no == 0 else (1e9,),
    #             result.error_no,
    #             result.all_cost,
    #             result.timestamp,
    #         ),
    #         "version": AUTOTVM_LOG_VERSION,
    #         "tvm_version": __version__,
    #     }
    #     return json.dumps(json_dict)

def read_json(file_path, op):
    inps =[]
    results = []
    with open(file_path, "r") as f:
        content = f.readlines()
        for line in content:
            if line.find(op)>=0:
                inp, result = decode(line, "json")
                inps.append(inp)
                results.append(result)
    return inps, results


def get_tiles_thread(inp, result):
    tiles = 0
    num_threads = inp.target.max_num_threads
    task_name = inp.task.name
    task_args = inp.task.args
    # thread_warp_size = inp.target.thread_warp_size
    costs = np.mean(result.costs)
    all_cost = result.all_cost
    config = inp.config
    configs = config._entity_map
    #calculate tiles
    tiles = 0
    for value in configs.values():
        # print(type(value))
        if(isinstance(value, task.space.SplitEntity)):
            tmp =1
            for i in value.size:
                if i!=1 and i!=-1:
                    tmp *= i
            tiles += tmp
    return {
        "task_name": task_name,
        "task_args":task_args,
        "tiles":tiles,
        "num_threads":num_threads,
        "costs":costs,
        "all_cost": all_cost,
    }

def get_op_name(op):
    start = op.find('"')
    end = op.find('"', start+1)
    return op[start+1:end]

if __name__ == '__main__':
    # parser = argparse.Argumentparser()
    # parser.add_argument("--modelname", type=str, default=none)
    # parser.add_argument("--tuner", type=str, default=none)
    # parser.add_argument("--target", type=str, default=none)
    # parser.add_argument("--op", type=str, default=none)
    # args = parser.parse_args()
    modelname = "bert"
    tuner = "xgb_knob"
    target = "cuda"
    op = "\"batch_matmul.cuda\", [[\"TENSOR\", [12, 128, 128], \"float32\"], [\"TENSOR\", [12, 64, 128], \"float32\"], [12, 128, 64], null, 0, 1], {}]"
    log = "/root/github/OpBench/data/Performance/"+modelname+ '-' + tuner +"-"+target+"-autotvm.json"
    tmp_log = log + ".tmp"
    inps, results = read_json(tmp_log, op)
    df = pd.DataFrame(columns=["trial","task_name", "task_args","tiles","num_threads","costs", "all_cost"])
    for i in range(len(inps)):
        if results[i].error_no == 0:
            item = get_tiles_thread(inps[i], results[i])
            item["trial"] = i
            df = df.append(item,ignore_index=True)
    op_name = get_op_name(op)
    df.to_csv("/root/github/OpBench/exp/tiles_thread/"+op_name+"_"+modelname+"_"+tuner+"_"+target+"_tiles_thread.csv", index=False)
            # res.costs
    # res.all_cost
    # res.error_no
    # res.timestamp
