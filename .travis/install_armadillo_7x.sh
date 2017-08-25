#!/usr/bin/env sh

wget http://sourceforge.net/projects/arma/files/armadillo-7.960.1.tar.xz
tar xvf armadillo-7.960.1.tar.xz
cd armadillo-7.960.1
mkdir build
cd build
cmake ..
make -j8
sudo make install
cd ..
cd ..
