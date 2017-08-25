#!/usr/bin/env sh

git clone https://github.com/google/googletest
cd googletest
mkdir build
cd build
cmake ..
make -j8
sudo make install
cd ..
cd ..
