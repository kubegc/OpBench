from tvm.autotvm.record import decode
import numpy as np
import pandas as pd
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
    # parser = argparse.Argumentparser()
    # parser.add_argument("--modelname", type=str, default=none)
    # parser.add_argument("--tuner", type=str, default=none)
    # parser.add_argument("--target", type=str, default=none)
    # args = parser.parse_args()
    modelname = "bert"
    tuner = "xgb_knob"
    target = "cuda"
    log = "/root/github/OpBench/data/Performance/"+modelname+ '-' + tuner +"-"+target+"-autotvm.json"
    tmp_log = log + ".tmp"
    inps, results = parse_logs(tmp_log)
    best_inps, best_results = parse_logs(log)
    df = pd.DataFrame(columns=["op_index", "task_name", "task_args","trial", "costs", "all_cost", "best_cost", "best_all_cost", "cmp"])
    for op_index in range(len(best_inps)):
        inp = best_inps[op_index]
        res = best_results[op_index]
        trial = 0
        # 筛选出一个算子的调优记录
        for i in range(len(inps)):
            tmp_inp = inps[i]
            tmp_res = results[i]
            if(inp.task.name==tmp_inp.task.name and inp.task.args == tmp_inp.task.args):
                trial += 1
                item = {
                   "op_index":op_index,
                   "task_name":tmp_inp.task.name,
                   "task_args": tmp_inp.task.args,
                   "trial":trial,
                   "costs": np.mean(tmp_res.costs),
                   "all_cost": tmp_res.all_cost,
                   "best_cost": np.mean(res.costs),
                   "best_all_cost": res.all_cost,
                    "cmp": (np.log(np.mean(tmp_res.costs)))/(np.log(np.mean(res.costs)))
                }
                if tmp_res.error_no == 0:
                    # item["costs"] = 0.0
                    # item["cmp"] = 0.0
                    df = df.append(item, ignore_index=True)
    print(df.size)
    df.to_csv("/root/github/OpBench/exp/op_costs/"+modelname+"_"+tuner+"_"+target+"_costs.csv", index=False)