# x86 or aarch64
architecture=$0
# python version 3.7,3.8
version=$1
if [ $architecture = "x86" -a $version = "3.8" ]; then
  wget -O Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
elif [ $architecture = "aarch64" -a $version = "3.8" ]; then
  wget -O Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-aarch64.sh
elif [ $architecture = "x86" -a $version = "3.7" ]; then
  wget -O Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh
elif [ $architecture = "aarch64" -a $version = "3.7" ]; then
  wget -O Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-aarch64.sh
fi

chmod 777 ./Miniconda3.sh
bash ./Miniconda3.sh
conda init
conda create -n python36 python=3.6
conda env list
conda activate python36
pip3 install psutil xgboost cloudpickle pretrainedmodels cython tensorflow transformers nni six numpy decorator attrs tornado onnx memory_profiler pytest
pip3 install --user mxnet requests "Pillow<7"
pip3 install torch==1.8.2+cpu torchvision==0.9.2+cpu torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html