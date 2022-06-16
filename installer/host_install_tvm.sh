# input tvm base PYTH: base_path=/root/github/as-tvm
base_path = $0
cd base_path
rm -rf build
mkdir build
cp cmake/config.cmake build
echo 'set(USE_VTA_FSIM ON)' >> build/config.cmake
echo 'set(USE_CUDA /usr/local/cuda)' >> build/config.cmake
echo 'set(USE_GRAPH_EXECUTOR ON)' >> build/config.cmake
echo 'set(USE_LLVM /usr/bin/llvm-config)' >> build/config.cmake
echo 'set(USE_PAPI ON)' >> build/config.cmake
cd build
cmake -DCMAKE_BUILD_TYPE=DEBUG ..
make -j40

export TVM_HOME=/root/github/as-tvm
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/vta/python:${PYTHONPATH}
export VTA_HW_PATH=$TVM_HOME/3rdparty/vta-hw

echo 'ready to test:'
python3 -m tvm.driver.tvmc --help
