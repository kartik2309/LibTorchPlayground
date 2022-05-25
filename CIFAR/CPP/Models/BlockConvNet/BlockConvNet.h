//
// Created by Kartik Rajeshwaran on 2022-05-19.
//

#ifndef LIBTORCHPLAYGROUND_CIFAR_MODELS_BLOCKCONVNET_BLOCKCONVNET_H_
#define LIBTORCHPLAYGROUND_CIFAR_MODELS_BLOCKCONVNET_BLOCKCONVNET_H_

#include <torch/torch.h>
#include <boost/log/trivial.hpp>

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
                                                       std::vector<int64_t> &stridesConv,
                                                       std::vector<int64_t> &dilationConv,
                                                       std::vector<int64_t> &kernelSizesPool,
                                                       std::vector<int64_t> &dilationPool,
                                                       std::vector<int64_t> &stridesPool);

  static std::string handle_path(std::string &path);

 public:
  // Constructor
  BlockConvNet(std::vector<int64_t> &imageDims,
               std::vector<int64_t> &channels,
               std::vector<int64_t> &kernelSizesConv,
               std::vector<int64_t> &stridesConv,
               std::vector<int64_t> &dilationConv,
               std::vector<int64_t> &kernelSizesPool,
               std::vector<int64_t> &stridesPool,
               std::vector<int64_t> &dilationPool,
               float_t dropout,
               int64_t numClasses);

  // Destructor
  ~BlockConvNet() override;

  // Forward Propagation
  torch::Tensor forward(torch::Tensor x);
  void save_model(std::string &path);
  void load_model(std::string &path);
};

#endif//LIBTORCHPLAYGROUND_CIFAR_MODELS_BLOCKCONVNET_BLOCKCONVNET_H_
