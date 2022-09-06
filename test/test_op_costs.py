from tvm.autotvm.record import decode
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, default=None)
    parser.add_argument("--tuner", type=str, default="random")
    parser.add_argument("--target", type=str, default="cuda")
    args = parser.parse_args()
    modelname = args.modelname
    tuner = args.tuner
    target = args.target
    #  python test/test_op_costs.py --modelname=resnet-18 --target=cuda --tuner=resnet-18
    # modelname = "resnet-18"
    # tuner = "xgb_knob"
    # target = "cuda"
    data_path = "/root/github/OpBench/exp/op_costs/"+modelname+"_"+tuner+"_"+target
    log = "/root/github/OpBench/data/Performance/"+modelname+ '-' + tuner +"-"+target+"-autotvm.json"
    tmp_log = log + ".tmp"
    inps, results = parse_logs(tmp_log)
    best_inps, best_results = parse_logs(log)
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.mkdir(data_path)
    
    for op_index in range(len(best_inps)):
        inp = best_inps[op_index]
        res = best_results[op_index]
        trial = 0
        df = pd.DataFrame(columns=[ "task_name", "trial", "costs", "all_cost", "best_cost", "best_all_cost", "cmp"])
        # 筛选出一个算子的调优记录
        for i in range(len(inps)):
            tmp_inp = inps[i]
            tmp_res = results[i]
            if(inp.task.name==tmp_inp.task.name and inp.task.args == tmp_inp.task.args):
                trial += 1
                item = {
                   "task_name":tmp_inp.task.name,
                   "trial":trial,
                   "costs": np.mean(tmp_res.costs),
                   "all_cost": tmp_res.all_cost,
                   "best_cost": np.mean(res.costs),
                   "best_all_cost": res.all_cost,
                    "cmp": (np.log(np.mean(tmp_res.costs)))/(np.log(np.mean(res.costs)))
                    # "cmp": np.exp(np.mean(res.costs)-tmp_res.costs)                
                    }
                if tmp_res.error_no == 0:
                    # item["costs"] = 0.0
                    # item["cmp"] = 0.0
                    df = df.append(item, ignore_index=True)
        print(df.size)
        worst_cost = df["costs"].max()
        best_cost = df["costs"].min()
        df["cmp"] = (worst_cost-df["costs"])/(worst_cost-best_cost)
        df.to_csv(data_path+"/"+str(op_index)+".csv",index=False)