//
// Created by Kartik Rajeshwaran on 2022-05-25.
//

#ifndef LIBTORCHPLAYGROUND_TORCHSCRIPTUTILITIES_CPP_LOADJIT_LOADJIT_H_
#define LIBTORCHPLAYGROUND_TORCHSCRIPTUTILITIES_CPP_LOADJIT_LOADJIT_H_

#include <torch/script.h>
#include <torch/torch.h>

class LoadJIT {

  // Variables
  torch::jit::Module model_;

 public:

  // Constructor
  LoadJIT(std::string &path);
  LoadJIT(const char* path);
  LoadJIT();

  // Destructor
  ~LoadJIT();

  // Function
  torch::jit::Module get_model();
  torch::jit::Module get_model(std::string &path);
  torch::jit::Module get_model(const char* path);

  torch::jit::Module* get_model_ptr();
  torch::jit::Module* get_model_ptr(std::string &path);
  torch::jit::Module* get_model_ptr(const char* path);

};

#endif//LIBTORCHPLAYGROUND_TORCHSCRIPTUTILITIES_CPP_LOADJIT_LOADJIT_H_
