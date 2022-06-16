cd /home/xilinx
git clone --recursive git@github.com:dos-lab/as-tvm.git tvm
cd /home/xilinx/tvm
mkdir build
cp cmake/config.cmake build/.
echo 'set(USE_VTA_FPGA ON)' >> build/config.cmake
cp 3rdparty/vta-hw/config/pynq_sample.json 3rdparty/vta-hw/config/vta_config.json
cmake ..
make runtime vta -j2

echo " wait 1 time, total 4 times"
make clean; make runtime vta -j2
echo " wait 2 time, total 4 times"
make clean; make runtime vta -j2
echo " wait 3 time, total 4 times"
make clean; make runtime vta -j2
echo " wait 4 time, total 4 times"
make clean; make runtime vta -j2

cd ..
sudo ./apps/vta_rpc/start_rpc_server.sh
