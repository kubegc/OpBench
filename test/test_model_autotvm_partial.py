## 从已有配置中抽取前100，前500.前1000个trail中的配置，并从中找出最优配置保存
from tvm.autotvm.record import decode, encode, pick_best
import numpy as np
import pandas as pd
import argparse
import shutil
import os
# 测量算子在100,500,1000此迭代中运行时间和最优时间的距离，假定3000次得到的时间为最优解
# batch_
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

def get_partial_ops(best_inps, best_results, inps, results, num):
    ret_inps =[]
    ret_results = []
    for op_index in range(len(best_inps)):
        inp = best_inps[op_index]
        res = best_results[op_index]
        cnt = 0
        for i in range(len(inps)):
            tmp_inp = inps[i]
            tmp_res = results[i]
            if(inp.task.name==tmp_inp.task.name and inp.task.args == tmp_inp.task.args):
                ret_inps.append(tmp_inp)
                ret_results.append(tmp_res)
                cnt += 1
            if(cnt >= num):
                break
        # print(inp)
        # print(cnt)
    return ret_inps, ret_results 

def get_partial_best(modelname, tuner, target, num):
    log = "/root/github/OpBench/data/Performance/"+modelname+ '-' + tuner +"-"+target+"-autotvm.json"
    tmp_log = log + ".tmp"
    inps, results = parse_logs(tmp_log)
    best_inps, best_results = parse_logs(log)
    partial_inps, partial_results = get_partial_ops(best_inps,best_results, inps, results, num)
    partial_log = "/root/github/OpBench/exp/partial_log/"+modelname+ '_' + tuner +"_"+target+"_"+str(num) +".json"
    partial_log_tmp = partial_log+".tmp" 
    with open(partial_log_tmp, "a") as f:
        for inp, result in zip(partial_inps, partial_results):
            f.write(encode(inp, result)+"\n")
    pick_best(partial_log_tmp, partial_log)      
    os.remove(partial_log_tmp)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser() 
    # parser.add_argument("--modelname", type=str, default=None)
    # parser.add_argument("--tuner", type=str, default="random")
    # parser.add_argument("--target", type=str, default="cuda")
    # parser.add_argument("--num", type=int, default=10)
    # args = parser.parse_args()
    # modelname = "bert"
    # tuner = "xgb_knob"
    # target = "cuda"
    # num = 500
    # modelname = args.modelname
    # tuner = args.tuner
    # target = args.target
    # num = args.num
    # models = ["inception_v3","mobilenet", "nasnetalarge", "roberta"]
    # tuners =["xgb_knob"]
    # targets = ["llvm","cuda"]
    # models = ["bert","resnet-18"]
    # tuners =["xgb_knob","random","grid","xgb"]
    # targets = ["cuda"]
    models = ["yolov3","resnet-18"]
    tuners =["xgb_knob"]
    targets = ["pynq"]
    nums = [100,300,500,1000]
    for modelname in models:
        for tuner in tuners:
            for target in targets:
                for num in nums: 
                    get_partial_best(modelname, tuner, target, num)
    