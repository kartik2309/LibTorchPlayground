
#include "CIFAR/CPP/cifar.hpp"

int main() {

  // Create Datasets
  std::string cifar_path_train = "/Users/kartikrajeshwaran/CodeSupport/CPP/Datasets/CIFAR-10-images/train";
  std::string cifar_path_test = "/Users/kartikrajeshwaran/CodeSupport/CPP/Datasets/CIFAR-10-images/test";

  auto *trainDataset = new CIFARTorchDataset(cifar_path_train);
  auto *evalDataset = new CIFARTorchDataset(cifar_path_test);

  // Create Models
  std::vector<int64_t> imageDims = {32, 32};

  std::vector<int64_t> channels = {3, 32, 64, 128};
  std::vector<int64_t> kernelSizesConv = {3, 3, 3};
  std::vector<int64_t> stridesConv = {1, 1, 1};
  std::vector<int64_t> dilationConv = {1, 1, 1};

  std::vector<int64_t> kernelSizesPool = {3, 3, 3};
  std::vector<int64_t> dilationPool = {1, 1, 1};
  std::vector<int64_t> stridesPool = {1, 1, 1};
  float_t dropout = 0.5;
  int64_t numClasses = 10;

  auto *blockConvModel = new BlockConvNet(imageDims,
                                          channels,
                                          kernelSizesConv,
                                          stridesConv,
                                          dilationConv,
                                          kernelSizesPool,
                                          stridesPool,
                                          dilationPool,
                                          dropout,
                                          numClasses);

  std::string save_path = "/Users/kartikrajeshwaran/CodeSupport/CPP/Models/LibtorchPlayground/BlockConvNet/";

  auto adamOptions = torch::optim::AdamWOptions(5e-3);
  torch::optim::AdamW adamWOptimizer(blockConvModel->parameters(), adamOptions);

  torch::nn::CrossEntropyLossOptions crossEntropyLossOptions = torch::nn::CrossEntropyLossOptions().reduction(torch::kMean);
  auto *crossEntropyLoss = new torch::nn::CrossEntropyLoss(crossEntropyLossOptions);

  auto *trainer = new Trainer<BlockConvNet *,
                              CIFARTorchDataset,
                              torch::data::transforms::Stack<>,
                              torch::data::samplers::SequentialSampler,
                              torch::optim::AdamW,
                              torch::nn::CrossEntropyLoss>(
      blockConvModel,
      *trainDataset,
      *evalDataset,
      32,
      adamWOptimizer,
      *crossEntropyLoss);

  trainer->fit(4);
  return 0;
}
