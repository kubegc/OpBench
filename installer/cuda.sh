version=$1
if [ $version = "11.0" ]; then
    wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
    sudo sh cuda_11.0.2_450.51.05_linux.run
else 
  echo "Unsupported Cuda Version"
  echo $version
fi