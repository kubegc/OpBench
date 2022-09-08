#!/bin/bash
# 收集模型在不同trial下结果的脚本

pyScriptPath="/root/github/OpBench/python/performance_collector/op_performance_collector.py"
resPath="/root/github/OpBench/exp/partial_res/"
# # resnet18 bert
# models=("resnet-18" "bert")
# tuners=("xgb_knob" "xgb" "grid" "random")
# targets=("cuda")
# modelsources=("local" "transformers")

# other models
models=("inception_v3" "mobilenet" "roberta" "nasnetalarge")
tuners=("xgb_knob")
targets=("llvm" "cuda")
modelsources=("local" "local" "transformers" "transformers")


# for model in ${models[*]} 
for((i=0; i<${#models[*]};i++))
do
  model=${models[$i]}
  modelsource=${modelsources[$i]}
  for tuner in ${tuners[*]}
  do
    for target in ${targets[*]}
    do
      echo "profile model ${model} modelsource ${modelsource} target ${target} tuner ${tuner}"
      python ${pyScriptPath} --modelsource=${modelsource} --modelname=${model} --ifcompare=true --tuner=${tuner} --ifpartial=true --target=${target}
    done
  done
done