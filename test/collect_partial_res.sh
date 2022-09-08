#!/bin/bash
# 收集模型在不同trial下结果的脚本
# $1 模型名称
# echo "test model $1 model source $2 on target $3"
pyScriptPath="/root/github/OpBench/python/performance_collector/op_performance_collector.py"
resPath="/root/github/OpBench/exp/partial_res/"
# models = ("inception_v3" "mobilenet" "nasnetalarge" "roberta")
# modelsources = ("local" "local" "transformers" "transformers")
models = ("inception_v3" "mobilenet" )
tuners = ("xgb_knob")
targets = ("llvm" "cuda")
modelsource = "local"
# tuners=("gridsearch" "random" "xgb" "xgb_knob")
# for tuner in ${tuners[*]}
# do
#   echo "profile tuner ${tuner}"
#   python ${pyScriptPath} --modelsource=$2 --modelname=$1 --ifcompare=true --tuner=$tuner --ifpartial=true --target=$3 > ${resPath}"$1-${tuner}-$3-res.txt"
# done
for model in ${models[*]} 
do
  for tuner in ${tuners[*]}
  do
    for target in ${targets[*]}
    do
      echo "profile model ${model} target ${target} tuner ${tuner}"
      python ${pyScriptPath} --modelsource=${modelsource} --modelname=${model} --ifcompare=true --tuner=${tuner} --ifpartial=true --target=${target}"
    done
  done
done



