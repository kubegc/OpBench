import os
import json
from unittest import result
import pandas as pd

def collect_tile_size(base_path, suffix):
    for root, dirs, files in os.walk(base_path):
        for name in files:
            if name.endswith(suffix):
                print(base_path+name)
                with open(base_path+name) as file:
                    while 1:
                        line = file.readline()
                        result = get_autotvm_tile_size(line)
                        if result > 0:
                            print(result)
                        if not line:
                            break

def get_autotvm_tile_size(line):
    if len(line) < 100:
        return -1
    log = json.loads(line)
    resut = 0
    for entity in log['config']['entity']:
        if isinstance(entity, list)  and len(entity) > 2 and entity[1] == 'sp' and isinstance(entity[2], list):
            temp_sum = 1
            for axis in entity[2]:
                if axis > 0:
                    temp_sum = temp_sum * axis
            resut = resut + temp_sum
    return resut

# collect_tile_size("/root/github/OpBench/data/Performance/", ".json")
def save_time_results(args,time_results ):
    if args.ifpartial is False:
        return
    dataPath = "/root/github/OpBench/exp/partial_res/"+args.target+".csv"
    if os.path.exists(dataPath):
        df = pd.read_csv(dataPath)
    else:
        df = pd.DataFrame(columns = ["modelname","tuner","0","100","300","500","1000","3000"])
    # print(time_results[0])
    # print(time_results[1])
    df = df.append({
        "modelname": args.modelname,
        "tuner":args.tuner,
        "0": time_results[0]["mean"],
        "100": time_results[2]["mean"],
        "300":time_results[3]["mean"],
        "500": time_results[4]["mean"],
        "1000":time_results[5]["mean"],
        "3000":time_results[1]["mean"],
    }, ignore_index=True)
    df.to_csv(dataPath, index=False)
    return 
    