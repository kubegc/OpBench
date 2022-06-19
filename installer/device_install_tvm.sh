cd /home/xilinx/tvm
rm -rf build
mkdir build
cp cmake/config.cmake build/.
echo 'set(USE_VTA_FPGA ON)' >> build/config.cmake
cp 3rdparty/vta-hw/config/pynq_sample.json 3rdparty/vta-hw/config/vta_config.json
cd build
cmake ..
make runtime vta -j2

echo " wait 1 time, total 4 times"
make clean; make runtime vta -j2

export TVM_HOME=/home/xilinx/tvm
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/vta/python:${PYTHONPATH}
export VTA_HW_PATH=$TVM_HOME/3rdparty/vta-hw
# python3 -m vta.exec.rpc_server --tracker=133.133.135.39:9190 --key=pynq
cd ..
sudo ./apps/vta_rpc/start_rpc_server.sh
