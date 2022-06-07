#include "Projects/CIFAR/CPP/Dataset/CIFARTorchDataset.h"
#include "Projects/CIFAR/CPP/Trainer/Trainer.hpp"
#include "TorchScriptUtilities/LoadJIT/LoadJIT.h"

int main() {
  std::string trainDatasetPath = "/Users/kartikrajeshwaran/CodeSupport/CPP/Datasets/CIFAR-10-images/train";
  std::string evalDatasetPath = "/Users/kartikrajeshwaran/CodeSupport/CPP/Datasets/CIFAR-10-images/test";
  std::string modelPath = "/Users/kartikrajeshwaran/CodeSupport/CPP/Models/LibtorchPlayground/BlockConvNet/BlockConvNetJIT.pt";

  auto *trainDataset = new CIFARTorchDataset(trainDatasetPath);
  auto *evalDataset = new CIFARTorchDataset(evalDatasetPath);

  auto loader = new LoadJIT(modelPath);
  auto model = loader->get_model_ptr();

  std::vector<torch::Tensor> params;
  for(auto i = model->parameters().begin(); i != model->parameters().end(); i++){
    params.push_back(*i);
  }

  torch::optim::AdamWOptions adamOptions(5e-3);
  torch::optim::AdamW adamOptimizer(params, adamOptions);

  torch::nn::CrossEntropyLoss *loss = new torch::nn::CrossEntropyLoss();

  auto *trainer = new Trainer<torch::jit::Module *,
                              CIFARTorchDataset,
                              torch::data::samplers::DistributedRandomSampler,
                              torch::optim::AdamW,
                              torch::nn::CrossEntropyLoss,
                              std::vector<torch::jit::IValue>>(
      model,
      *trainDataset,
      *evalDataset,
      32,
      adamOptimizer,
      *loss,
      4);

  trainer->fit_parallel(16, 0);
  return 0;
}
