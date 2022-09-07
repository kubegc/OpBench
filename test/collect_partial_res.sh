#!/bin/bash
# 收集模型在不同trial下结果的脚本
# $1 模型名称
echo "test model $1 model source $2 on target $3"
pyScriptPath="/root/github/OpBench/python/performance_collector/op_performance_collector.py"
resPath="/root/github/OpBench/exp/partial_res/"

tuners=("gridsearch" "random" "xgb" "xgb_knob")
# for ((i=0; i<${#tuners[@]};i++));
for tuner in ${tuners[*]}
do
  echo "profile tuner ${tuner}"
  python ${pyScriptPath} --modelsource=$2 --modelname=$1 --ifcompare=true --tuner=$tuner --ifpartial=true --target=$3 > ${resPath}"$1-${tuner}-$3-res.txt"
done
