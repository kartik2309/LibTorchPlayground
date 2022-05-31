
#include "ONNX/Inference/Inference.h"
#include "Projects/CIFAR/CPP/cifar.hpp"
#include "TorchScriptUtilities/CPP/LoadJIT/LoadJIT.h"
#include "Trainer/CPP/Trainer.hpp"

int main() {

  //   Create Datasets
  std::string cifar_path_train = "/Users/kartikrajeshwaran/CodeSupport/CPP/Datasets/CIFAR-10-images/train";
  std::string cifar_path_test = "/Users/kartikrajeshwaran/CodeSupport/CPP/Datasets/CIFAR-10-images/test";
  std::string model_path = "/Users/kartikrajeshwaran/CodeSupport/CPP/Models/LibtorchPlayground/BlockConvNet/BlockConvNet.onnx";

  auto *trainDataset = new CIFARTorchDataset(cifar_path_train);
  auto *evalDataset = new CIFARTorchDataset(cifar_path_test);
  {

//  auto *loadJit = new LoadJIT(model_path);
//  auto jitModel = loadJit->get_model_ptr();
//
//  std::vector<torch::Tensor> params;
//  for (auto param : jitModel->parameters()){
//    params.push_back(param);
//  }
//
//  auto adamOptions = torch::optim::AdamWOptions(5e-3);
//  torch::optim::AdamW adamWOptimizer(params, adamOptions);
//
//  torch::nn::CrossEntropyLossOptions crossEntropyLossOptions = torch::nn::CrossEntropyLossOptions().reduction(torch::kMean);
//  auto *crossEntropyLoss = new torch::nn::CrossEntropyLoss(crossEntropyLossOptions);
//
//
    //  auto *trainer = new Trainer<torch::jit::Module *,
    //                              CIFARTorchDataset,
    //                              torch::data::samplers::SequentialSampler,
    //                              torch::optim::AdamW,
    //                              torch::nn::CrossEntropyLoss,
    //                              std::vector<torch::jit::IValue>>(
    //      jitModel,
    //      *trainDataset,
    //      *evalDataset,
    //      32,
    //      adamWOptimizer,
    //      *crossEntropyLoss);
    //
    //  trainer->fit(7);
    //  jitModel->to(torch::kCPU);
    //  jitModel->save(model_path);
  }

  std::vector<std::string> inputNames = {"images"};
  std::vector<std::string> outputNames = {"logits"};
  Inference *onnxInference = new Inference(model_path, inputNames, outputNames);
  torch::Tensor image = trainDataset->get(0).data.unsqueeze(0);
  onnxInference->run_inference(image);
  return 0;
}
