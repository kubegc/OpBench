from tvm import autotvm

with autotvm.apply_history_best("resnet-18.log") as ab:
    print(ab.best_by_model)
    print(ab.best_by_targetkey)
    print(ab._best_user_defined)

with autotvm.apply_history_best("/root/github/OpBench/data/Performance/resnet-18-grid-cuda-autotvm.json") as ab:
    print(ab.best_by_model)
    print(ab.best_by_targetkey)
    print(ab._best_user_defined)

