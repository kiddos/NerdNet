Nerual Network
==============

[![Build Status](https://travis-ci.org/kiddos/nnet.svg?branch=master)](https://travis-ci.org/kiddos/nnet)


## Dependencies

In order to build examples, optional dependencies need to be installed

* [Armadillo](http://arma.sourceforge.net/)
* [OpenBLAS](http://www.openblas.net/), ([github link](https://github.com/xianyi/OpenBLAS)) (Optional)
* [OpenCV](http://opencv.org/) 3.x, ([github link](https://github.com/opencv/opencv)) (Optional)
* [MathGL](http://mathgl.sourceforge.net/doc_en/Main.html) (Optional)

### Installing dependencies (Ubuntu 16.04)

* Armadillo

```shell
sudo apt-get install libarmadillo-dev
```

* OpenBLAS

```shell
sudo apt-get install libopenblas-dev
```

* OpenCV

```shell
git clone -b 3.2.0 https://github.com/opencv/opencv
cd opencv
mkdir build
cd build
cmake -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j
sudo make install
```

* MathGL

```shell
sudo apt-get install libmgl-dev
```

## Build

```shell
mkdir build
cmake ..
make -j
```

### Run Test

```shell
make test
```
