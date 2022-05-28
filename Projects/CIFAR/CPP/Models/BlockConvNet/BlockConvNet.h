//
// Created by Kartik Rajeshwaran on 2022-05-19.
//

#ifndef LIBTORCHPLAYGROUND_PROJECTS_CIFAR_CPP_MODELS_BLOCKCONVNET_BLOCKCONVNET_H_
#define LIBTORCHPLAYGROUND_PROJECTS_CIFAR_CPP_MODELS_BLOCKCONVNET_BLOCKCONVNET_H_
#define BOOST_LOG_DYN_LINK 1

#include <boost/log/trivial.hpp>
#include <torch/torch.h>
#include <filesystem>

class BlockConvNet : public torch::nn::Module {

 private:
  // Variables
  struct ModuleArgs {
    torch::Tensor imageDims_;
    torch::Tensor channels_;
    torch::Tensor kernelSizesConv_;
    torch::Tensor stridesConv_;
    torch::Tensor dilationConv_;
    torch::Tensor kernelSizesPool_;
    torch::Tensor stridesPool_;
    torch::Tensor dilationPool_;
    torch::Tensor dropout_;
    torch::Tensor numClasses_;

    ModuleArgs(torch::Tensor &imageDims,
               torch::Tensor &channels,
               torch::Tensor &kernelSizesConv,
               torch::Tensor &stridesConv,
               torch::Tensor &dilationConv,
               torch::Tensor &kernelSizesPool,
               torch::Tensor &stridesPool,
               torch::Tensor &dilationPool,
               torch::Tensor &dropout,
               torch::Tensor &numClasses);
    ModuleArgs();
  } moduleArgs;

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

  void setup_model(std::vector<int64_t> &imageDims,
                   std::vector<int64_t> &channels,
                   std::vector<int64_t> &kernelSizesConv,
                   std::vector<int64_t> &stridesConv,
                   std::vector<int64_t> &dilationConv,
                   std::vector<int64_t> &kernelSizesPool,
                   std::vector<int64_t> &stridesPool,
                   std::vector<int64_t> &dilationPool,
                   float_t dropout,
                   int64_t numClasses);

  static std::string handle_path(std::string &path);
  static std::vector<int64_t> inverse_stack(torch::Tensor &tensor);

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

  BlockConvNet(std::string &path, std::optional<float_t> dropout_ = std::optional<float_t>());

  // Destructor
  ~BlockConvNet() override;

  // Forward Propagation
  torch::Tensor forward(torch::Tensor x);
  void save_model(std::string &path);
  void load_model(std::string &path);
};

#endif//LIBTORCHPLAYGROUND_PROJECTS_CIFAR_CPP_MODELS_BLOCKCONVNET_BLOCKCONVNET_H_
