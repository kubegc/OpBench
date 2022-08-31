from html import entities
import pandas as pd
import json
from tvm.autotvm.record import decode

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

def read_json(file_path):
    inps =[]
    results = []
    with open(file_path, "r") as f:
        content = f.readlines()
        for line in content:
            # res.append(json.loads(line))
            inp, result = decode(line, "json")
            inps.append(inp)
            results.append(result)
    return inps, results


def parse_logs(logs, op):
    """
    获取算子名称对应log
    """
    pass

if __name__ == '__main__':
    modelname = "resnet-18"
    tuner = "xgb"
    target = "cuda"
    log = "/root/github/OpBench/data/Performance/"+modelname+ '-' + tuner +"-"+target+"-autotvm.json"
    tmp_log = log + ".tmp"
    inps, results = read_json(log)
    tiles = 0 #线程数量
    for i in range(len(inps)):
        inp =  inps[i]
        result = results[i]

        task_name = inp.task.name
        task_args = inp.task.args
        num_threads = inp.target.max_num_threads
        thread_warp_size = inp.target.thread_warp_size
        config = inp.config
        configs = config._entity_map
        for value in configs.values():
            tmp =1
            for i in value:
                if i!=1 and i!=-1:
                    tmp *= i
            tiles += tmp
        
    
    # res.costs
    # res.all_cost
    # res.error_no
    # res.timestamp
    print(len(inps))
    print(inps[0])
    print("test")