from html import entities
import pandas as pd
import numpy as np
import json
from tvm.autotvm.record import decode
from tvm.autotvm import task
import argparse
import os
import shutil

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

def parse_logs(file_path):
    inps =[]
    results = []
    with open(file_path, "r") as f:
        content = f.readlines()
        for line in content:
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
    return tiles, num_threads
    # return {
    #     "task_name": task_name,
    #     "task_args":task_args,
    #     "tiles":tiles,
    #     "num_threads":num_threads,
    #     "costs":costs,
    #     "all_cost": all_cost,
    # }

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
    # op = "\"batch_matmul.cuda\", [[\"TENSOR\", [12, 128, 128], \"float32\"], [\"TENSOR\", [12, 64, 128], \"float32\"], [12, 128, 64], null, 0, 1], {}]"
    data_path = "/root/github/OpBench/exp/tiles_thread/"+modelname+"_"+tuner+"_"+target
    log = "/root/github/OpBench/data/Performance/"+modelname+ '-' + tuner +"-"+target+"-autotvm.json"
    tmp_log = log + ".tmp"
    num = 20
    best_inps, best_results = parse_logs(log)
    inps, results = parse_logs(tmp_log)

    #创建针对一个模型的文件夹
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.mkdir(data_path)

    for op_index in range(len(best_inps)):
        inp = best_inps[op_index]
        res = best_results[op_index]
        print("process " + inp.task.name)
        trial = 0
        # 收集一个算子对应的record
        tmp_inps = []
        tmp_results = []
        df = pd.DataFrame(columns=["tiles", "num_threads", "costs","best_costs", "all_cost", "log"])
        for i in range(len(inps)):
            tmp_inp = inps[i]
            tmp_res = results[i]
            if(inp.task.name==tmp_inp.task.name and inp.task.args == tmp_inp.task.args): 
                tmp_inps.append(tmp_inp)
                tmp_results.append(tmp_res)   
        trial = 0
        for i in range(len(tmp_inps)//num):
            inp_slice = tmp_inps[i*num:(i+1)*num]
            res_slice = tmp_results[i*num:(i+1)*num]
            best_trail = trial
            inp_slice_best = inp_slice[0]
            res_slice_best = res_slice[0]
            for tmp_inp, tmp_res in zip(inp_slice, res_slice):
                trial += 1
                if tmp_res.error_no == 0 and np.mean(tmp_res.costs) < np.mean(res_slice_best.costs): 
                    inp_slice_best = tmp_inp
                    res_slice_best = tmp_res
                    best_trail = trial
            if not res_slice_best.error_no == 0:
                continue
            tiles, num_threads = get_tiles_thread(inp_slice_best, res_slice_best)
            item = {
                "trial": best_trail,
                "tiles": tiles,
                "num_threads": num_threads,
                "costs": np.mean(res_slice_best.costs),
                "best_costs": np.mean(res.costs),
                "all_cost": res_slice_best.all_cost,
                "log":(np.log(np.mean(res_slice_best.costs)))/(np.log(np.mean(res.costs))),
            }    
            df = df.append(item,ignore_index=True)
        df.to_csv(data_path+"/"+str(op_index)+".csv",index=False)    
            # res.costs
    # res.all_cost
    # res.error_no
    # res.timestamp
