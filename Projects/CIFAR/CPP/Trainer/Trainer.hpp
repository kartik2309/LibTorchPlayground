//
// Created by Kartik Rajeshwaran on 2022-05-23.
//

#ifndef LIBTORCHPLAYGROUND_TRAINER_CPP_TRAINER_HPP_
#define LIBTORCHPLAYGROUND_TRAINER_CPP_TRAINER_HPP_
#define BOOST_LOG_DYN_LINK 1

#include <boost/log/trivial.hpp>
#include <torch/torch.h>
#include <vector>

#include "../../../../Trainer/TrainerBase.hpp"
#include "../../../../Trainer/Utilities/BatchTransformTemplate.hpp"
#include "../../../../Trainer/Utilities/Metrics/Metrics.h"
#include "../../../../Trainer/Utilities/MultiprocessingBackend.hpp"

using namespace torch::indexing;

template<class TorchModule,
         class TorchDataset,
         class Sampler,
         class Optimizer,
         class Loss,
         class ReturnType = torch::Tensor>
class Trainer : public TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType> {

  // Functions
  void train_loop() override;
  void eval_loop() override;
  void test_loop() override;

 public:
  // Constructor
  Trainer(TorchModule &model,
          TorchDataset &trainDataset,
          TorchDataset &evalDataset,
          int batchSize,
          Optimizer &optimizer,
          Loss &loss);

  Trainer(TorchModule &model,
          TorchDataset &trainDataset,
          int batchSize,
          Optimizer &optimizer,
          Loss &loss);

  Trainer(TorchModule &model,
          TorchDataset &trainDataset,
          TorchDataset &evalDataset,
          TorchDataset &testDataset,
          int batchSize,
          Optimizer &optimizer,
          Loss &loss);

  Trainer(TorchModule &model,
          TorchDataset &trainDataset,
          TorchDataset &evalDataset,
          int batchSize,
          Optimizer &optimizer,
          Loss &loss,
          int n_procs);

  // Destructor
  ~Trainer();
};

// ------------------------------------------ DEFINITIONS ------------------------------------------------------

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
void Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::train_loop() {
  std::vector<float> losses;
  std::vector<float> accuracies;

  this->model_->train();
  for (auto &batch : *this->trainDataLoader_) {

    auto data = batch.data;
    auto target = batch.target;

    torch::Tensor logits = this->to_tensor(this->model_->forward(data));
    torch::Tensor predictedClasses = torch::argmax(logits, -1);
    torch::Tensor loss = this->train_step(logits, target);

    float accuracyValue = Metrics::accuracy(predictedClasses, target);

    losses.push_back(loss.item<float>());
    accuracies.push_back(accuracyValue);
  }
  std::vector<float> trainMetrics = Metrics::get_batch_metrics(losses, accuracies);
  BOOST_LOG_TRIVIAL(info) << "Loss:" << trainMetrics[0] << " Accuracy:" << trainMetrics[1];
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
void Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::eval_loop() {
  std::vector<float> losses;
  std::vector<float> accuracies;

  this->model_->eval();
  torch::NoGradGuard noGradGuard;
  {
    for (auto &batch : *this->evalDataLoader_) {
      auto data = batch.data;
      torch::Tensor target = batch.target;

      auto logits = this->to_tensor(this->model_->forward(data));
      torch::Tensor predictedClasses = torch::argmax(logits, -1);

      torch::Tensor loss = this->eval_step(logits, target);
      float accuracyValue = Metrics::accuracy(predictedClasses, target);

      losses.push_back(loss.item<float>());
      accuracies.push_back(accuracyValue);
    }
  }
  std::vector<float> evalMetrics = Metrics::get_batch_metrics(losses, accuracies);
  BOOST_LOG_TRIVIAL(info) << "Loss:" << evalMetrics[0] << " Accuracy:" << evalMetrics[1];
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
void Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::test_loop() {
  std::vector<float> losses;
  std::vector<float> accuracies;

  this->model_->eval();
  torch::NoGradGuard noGradGuard;
  {
    for (auto &batch : *this->testDataLoader_) {
      auto data = batch.data;
      torch::Tensor target = batch.target;

      auto logits = this->to_tensor(this->model_->forward(data));
      torch::Tensor predictedClasses = torch::argmax(logits, -1);

      torch::Tensor loss = this->eval_step(logits, target);
      float accuracyValue = Metrics::accuracy(predictedClasses, target);

      losses.push_back(loss.item<float>());
      accuracies.push_back(accuracyValue);
    }
  }
  std::vector<float> testMetrics = Metrics::get_batch_metrics(losses, accuracies);
  BOOST_LOG_TRIVIAL(info) << "Loss:" << testMetrics[0] << " Accuracy:" << testMetrics[1];
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::Trainer(
    TorchModule &model,
    TorchDataset &trainDataset,
    TorchDataset &evalDataset,
    int batchSize,
    Optimizer &optimizer,
    Loss &loss) : TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>(model,
                                                                                               trainDataset,
                                                                                               evalDataset,
                                                                                               batchSize,
                                                                                               optimizer,
                                                                                               loss) {
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::Trainer(
    TorchModule &model,
    TorchDataset &trainDataset,
    int batchSize,
    Optimizer &optimizer,
    Loss &loss) : TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>(model,
                                                                                               trainDataset,
                                                                                               batchSize,
                                                                                               optimizer,
                                                                                               loss) {
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::Trainer(
    TorchModule &model,
    TorchDataset &trainDataset,
    TorchDataset &evalDataset,
    TorchDataset &testDataset,
    int batchSize,
    Optimizer &optimizer,
    Loss &loss) : TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>(model,
                                                                                               trainDataset,
                                                                                               evalDataset,
                                                                                               testDataset,
                                                                                               batchSize,
                                                                                               optimizer,
                                                                                               loss) {
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::Trainer(
    TorchModule &model,
    TorchDataset &trainDataset,
    TorchDataset &evalDataset,
    int batchSize,
    Optimizer &optimizer,
    Loss &loss,
    int n_procs) : TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>(model,
                                                                                                trainDataset,
                                                                                                evalDataset,
                                                                                                batchSize,
                                                                                                optimizer,
                                                                                                loss,
                                                                                                n_procs){

}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::~Trainer() = default;

#endif//LIBTORCHPLAYGROUND_TRAINER_CPP_TRAINER_HPP_