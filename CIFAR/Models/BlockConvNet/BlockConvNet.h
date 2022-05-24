//
// Created by Kartik Rajeshwaran on 2022-05-19.
//

#ifndef LIBTORCHPLAYGROUND_CIFAR_CIFAR_H_
#define LIBTORCHPLAYGROUND_CIFAR_CIFAR_H_

#include <torch/torch.h>

class BlockConvNet : public torch::nn::Module {

 private:
  // Variables
  torch::nn::ModuleList convSubmodules = torch::nn::ModuleList();
  torch::nn::ModuleList maxPoolSubmodules = torch::nn::ModuleList();

  torch::nn::Flatten *flattenSubmodule = nullptr;
  torch::nn::Dropout *dropoutSubmodule = nullptr;
  torch::nn::Linear *linearSubmodule = nullptr;

  torch::nn::ReLU relu = torch::nn::ReLU();

  // Functions
  static std::vector<std::vector<int64_t>> get_interim(std::vector<int64_t> &imageDims,
                                                       std::vector<int64_t> &kernelSizesConv,
                                                       std::vector<int64_t> &strides,
                                                       std::vector<int64_t> &dilation,
                                                       std::vector<int64_t> &kernelSizesPool,
                                                       std::vector<int64_t> &dilationPool,
                                                       std::vector<int64_t> &stridesPool);

 public:
  // Constructor
  BlockConvNet(std::vector<int64_t> &imageDims,
               std::vector<int64_t> &channels,
               std::vector<int64_t> &kernelSizesConv,
               std::vector<int64_t> &strides,
               std::vector<int64_t> &dilation,
               std::vector<int64_t> &kernelSizesPool,
               std::vector<int64_t> &dilationPool,
               std::vector<int64_t> &stridesPool,
               float_t dropout,
               int64_t numClasses);

  // Destructor
  ~BlockConvNet() override;

  // Forward Propagation
  torch::Tensor forward(torch::Tensor x);
};

#endif//LIBTORCHPLAYGROUND_CIFAR_CIFAR_H_
