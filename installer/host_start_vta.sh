# python36
# env
source install/env_var.sh

# start tracker on host
python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190

# register device to rpc tracker(root user)
# need set environment(device_install_tvm.sh)
python3 -m vta.exec.rpc_server --tracker=hostIP:9190 --key=pynq

# query the rpc tracker to check if the device is ready
python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190