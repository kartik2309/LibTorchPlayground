# LibTorchPlayground
## Introduction

A collection of Libtorch Library based implementations with corresponding Pytorch (Python) based implementation.

Models can be prototyped in Python and exported to JIT and trained in C++ with CUDA or CPU - Multiprocessing training has
been implemented with MPI and currently works only on CPU.

Trainer is a generic class in C++ (`Trainer/CPP/Trainer/Trainer.h`) that can work with `jit::Module` and `nn::Module` in order to perform training and inference.
Python trainer `Trainer/Python/trainer.py` can be used to perform training and inference for a model but more importantly it can be used for prototyping and exporting 
models to `JIT` via `TorchScript`. The exported model then can be trained with C++ Trainer. Additionally Trainer C++ can be inherited if one wishes to add 
additional features as per their use-case. Trainer supports CUDA training and inference. 


Toy Implementations included:

- CIFAR: Simple implementation for CIFAR dataset, includes files for dataset preparation and models.

## Dependencies
- `libtorch`
- `onnxruntime`
- `boost/log`
- `boost/mpi`
- `opencv`
- `openmpi`
- `openmp`

## To Build

To Build with CUDA
```
mkdir -p LibTorchPlayground-build
cd LibTorchPlayground-build
cmake -DCMAKE_PREFIX_PATH="/path/to/libtorch" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CXX_STANDARD_REQUIRED=ON -Wl,--whole-archive on libc10_cuda.a ../LibTorchPlayground/
cmake --build . --config Release
make
```
To Build with CPU
```
mkdir -p LibTorchPlayground-build
cd LibTorchPlayground-build
cmake -DCMAKE_PREFIX_PATH="/path/to/libtorch" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CXX_STANDARD_REQUIRED=ON ../LibTorchPlayground/
cmake --build . --config Release
make
```

You may need to pass path to `onnxruntime` build via `-DCMAKE_PREFIX_PATH` if it's not installed via package managers. 
You can skip adding argument `-DCMAKE_PREFIX_PATH` for cmake if you have libtorch, onnxruntime and other packages installed via package manager (homebrew for example).  

To run with mpi with 4 processes
```
cd LibTorchPlayground-build
mpirun -n 4 LibTorchPlayground
```