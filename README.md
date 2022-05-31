# LibTorchPlayground
## Introduction

A collection of Libtorch Library based implementations with corresponding Pytorch (Python) based implementation.

Models can be prototyped in Python and exported to JIT and trained in C++ with CUDA or CPU.

Trainer is a generic class in C++ (`Trainer/CPP/Trainer/Trainer.h`) that can work with `jit::Module` and `nn::Module` in order to perform training and inference.
Python trainer `Trainer/Python/trainer.py` can be used to perform training and inference for a model but more importantly it can be used for prototyping and exporting 
models to `JIT` via `TorchScript`. The exported model then can be trained with C++ Trainer. Additionally Trainer C++ can be inherited if one wishes to add 
additional features as per their use-case. Trainer supports CUDA training and inference. 


Toy Implementations included:

- CIFAR: Simple implementation for CIFAR dataset, includes files for dataset preparation and models.

## Dependencies


## To Build
