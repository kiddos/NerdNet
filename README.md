Nerual Network
==============

[![Build Status](https://travis-ci.org/kiddos/NerdNet.svg?branch=master)](https://travis-ci.org/kiddos/NerdNet)


## Dependencies

In order to build examples, optional dependencies need to be installed

* [Armadillo](http://arma.sourceforge.net/)
* [Boost](http://www.boost.org/)
* [OpenBLAS](http://www.openblas.net/), ([github link](https://github.com/xianyi/OpenBLAS)) (Optional)
* [Qt](https://www.qt.io/) 5.x

### Installing dependencies (Ubuntu 16.04)

* Armadillo

```shell
sudo apt-get install libarmadillo-dev
```

* Boost

```shell
sudo apt-get install libboost-all-dev
```

* OpenBLAS

```shell
sudo apt-get install libopenblas-dev
```

* Qt components

```shell
sudo apt-get install libqt5widget5 libqt5printsupport5 libqcustomplot
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


## Reference

Armadillo C++ Linear Algebra Library
Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
Copyright 2008-2016 National ICT Australia (NICTA)

This product includes software developed by Conrad Sanderson (http://conradsanderson.id.au)
This product includes software developed at National ICT Australia (NICTA)
