#include <fstream>
#include <iostream>

#include "CIFAR/Dataset/CIFARTorchDataset.h"
#include "CIFAR/Models/BlockConvNet/BlockConvNet.h"
#include "CIFAR/Trainer/Trainer.h"

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
  std::vector<int64_t> strides = {1, 1, 1};
  std::vector<int64_t> dilation = {1, 1, 1};

  std::vector<int64_t> kernelSizesPool = {3, 3, 3};
  std::vector<int64_t> dilationPool = {1, 1, 1};
  std::vector<int64_t> stridesPool = {1, 1, 1};
  float_t dropout = 0.5;
  int64_t numClasses = 10;

  auto *blockConvModel = new BlockConvNet(imageDims,
                                          channels,
                                          kernelSizesConv,
                                          strides,
                                          dilation,
                                          kernelSizesPool,
                                          dilationPool,
                                          stridesPool,
                                          dropout,
                                          numClasses);


  auto *trainer = new Trainer<BlockConvNet*, CIFARTorchDataset, torch::data::transforms::Stack<>, torch::data::samplers::SequentialSampler>(
      blockConvModel,
      *trainDataset,
      *evalDataset,
      32,
      5e-3);
  trainer->fit(1);

  return 0;
}
