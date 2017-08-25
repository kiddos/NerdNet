#!/usr/bin/env sh

sudo apt-get install git -y
git clone https://github.com/google/benchmark
cd benchmark
mkdir build
cd build
cmake ..
make -j8
sudo make install
cd ..
