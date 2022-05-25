//
// Created by Kartik Rajeshwaran on 2022-05-23.
//

#ifndef LIBTORCHPLAYGROUND_CIFAR_TRAINER_TRAINER_H_
#define LIBTORCHPLAYGROUND_CIFAR_TRAINER_TRAINER_H_

#include <torch/torch.h>
#include <boost/log/trivial.hpp>

using namespace torch::indexing;

template<class TorchModule, class TorchDataset, class DatasetTransformation, class Sampler, class Optimizer, class Loss>
class Trainer {

  // Variables
  typedef torch::disable_if_t<
      torch::data::datasets::MapDataset<
          TorchDataset, DatasetTransformation>::is_stateful
          || !std::is_constructible<Sampler, size_t>::value,
      std::unique_ptr<torch::data::StatelessDataLoader<
          torch::data::datasets::MapDataset<TorchDataset, DatasetTransformation>, Sampler>>>
      TorchDataLoader;

  TorchModule model_;
  Optimizer *optimizer_;
  Loss *loss_;
  TorchDataLoader trainDataLoader_;
  TorchDataLoader evalDataLoader_;

  // Functions
  std::vector<float> train_loop();
  std::vector<float> eval_loop();

  float accuracy(torch::Tensor &predictedClasses, torch::Tensor &trueClasses);
  std::vector<float> get_batch_metrics(std::vector<float> &losses, std::vector<float> &accuracies);

  TorchDataLoader setup_data_loader(TorchDataset &dataset, int batchSize);

 public:
  // Constructor
  Trainer(TorchModule &model,
          TorchDataset &trainDataset,
          TorchDataset &evalDataset,
          int batchSize,
          Optimizer &optimizer,
          Loss &loss);

  // Destructor
  ~Trainer();

  // Functions
  void fit(int epochs);
};

#endif//LIBTORCHPLAYGROUND_CIFAR_TRAINER_TRAINER_H_
