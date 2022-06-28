import os
import json
from unittest import result

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

collect_tile_size("/root/github/OpBench/data/Performance/", ".json")