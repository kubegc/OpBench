bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

sudo apt-get install llvm-10*
ln -s /usr/bin/llvm-config-10 /usr/bin/llvm-config
/usr/bin/llvm-config --version