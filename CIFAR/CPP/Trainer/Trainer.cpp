//
// Created by Kartik Rajeshwaran on 2022-05-23.
//

#include "Trainer.h"

template<class TorchModule, class TorchDataset, class DatasetTransformation, class Sampler, class Optimizer, class Loss>
std::vector<float> Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler, Optimizer, Loss>::train_loop() {
  std::vector<float> losses;
  std::vector<float> accuracies;

  model_->train();
  for (torch::data::Example<> &batch : *trainDataLoader_) {
    torch::Tensor data = batch.data;
    torch::Tensor target = batch.target;

    data = model_->forward(data);
    torch::Tensor predictedClasses = torch::argmax(data, -1);

    optimizer_->zero_grad();
    torch::Tensor loss = loss_->ptr()->forward(data, target);
    loss.backward();
    optimizer_->step();

    float accuracyValue = accuracy(predictedClasses, target);

    losses.push_back(loss.item<float>());
    accuracies.push_back(accuracyValue);
  }

  return get_batch_metrics(losses, accuracies);
}

template<class TorchModule, class TorchDataset, class DatasetTransformation, class Sampler, class Optimizer, class Loss>
std::vector<float> Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler, Optimizer, Loss>::eval_loop() {
  std::vector<float> losses;
  std::vector<float> accuracies;

  model_->eval();
  for (torch::data::Example<> &batch : *evalDataLoader_) {
    torch::Tensor data = batch.data;
    torch::Tensor target = batch.target;

    data = model_->forward(data);
    torch::Tensor predictedClasses = torch::argmax(data, -1);

    torch::Tensor loss = loss_->ptr()->forward(data, target);
    float accuracyValue = accuracy(predictedClasses, target);

    losses.push_back(loss.item<float>());
    accuracies.push_back(accuracyValue);
  }

  return get_batch_metrics(losses, accuracies);
}

template<class TorchModule, class TorchDataset, class DatasetTransformation, class Sampler, class Optimizer, class Loss>
float Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler, Optimizer, Loss>::accuracy(torch::Tensor &predictedClasses, torch::Tensor &trueClasses) {
  torch::Tensor ones_ = torch::ones(predictedClasses.sizes());
  torch::Tensor onesIndexed = ones_.index({predictedClasses.eq(trueClasses)});
  torch::Tensor mean_ = onesIndexed.sum() / ones_.sum();
  return mean_.item<float>();
}

template<class TorchModule, class TorchDataset, class DatasetTransformation, class Sampler, class Optimizer, class Loss>
std::vector<float> Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler, Optimizer, Loss>::get_batch_metrics(std::vector<float> &losses,
                                                                                                                          std::vector<float> &accuracies) {
  float totalLoss = 0.00;
  for (auto &value : losses) {
    totalLoss += value;
  }
  float avgLoss = totalLoss / (float) losses.size();

  float totalAccuracy = 0.00;
  for (auto &value : accuracies) {
    totalAccuracy += value;
  }
  float avgAccuracy = totalAccuracy /(float) accuracies.size();

  std::vector<float> metrics = {avgLoss, avgAccuracy};
  return metrics;
}

template<class TorchModule, class TorchDataset, class DatasetTransformation, class Sampler, class Optimizer, class Loss>
Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler, Optimizer, Loss>::Trainer(TorchModule &model,
                                                                                             TorchDataset &trainDataset,
                                                                                             TorchDataset &evalDataset,
                                                                                             int batchSize,
                                                                                             Optimizer &optimizer,
                                                                                             Loss &loss) {

  model_ = model;
  optimizer_ = &optimizer;
  loss_ = &loss;
  trainDataLoader_ = setup_data_loader(trainDataset, batchSize);
  evalDataLoader_ = setup_data_loader(evalDataset, 1);
}

template<class TorchModule, class TorchDataset, class DatasetTransformation, class Sampler, class Optimizer, class Loss>
Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler, Optimizer, Loss>::~Trainer() = default;

template<class TorchModule, class TorchDataset, class DatasetTransformation, class Sampler, class Optimizer, class Loss>
void Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler, Optimizer, Loss>::fit(int epochs) {

  for (int idx = 0; idx != epochs; idx++) {
    BOOST_LOG_TRIVIAL(info) << "Epoch:" << idx;
    BOOST_LOG_TRIVIAL(info) << "Training Phase";
    std::vector<float> trainMetrics = train_loop();
    BOOST_LOG_TRIVIAL(info) << "Loss:" << trainMetrics[0] << " Accuracy:" << trainMetrics[1];

    BOOST_LOG_TRIVIAL(info) << "Evaluation Phase";
    std::vector<float> evalMetrics = eval_loop();
    BOOST_LOG_TRIVIAL(info) << "Loss:" << evalMetrics[0] << " Accuracy:" << evalMetrics[1] << std::endl;
  }
}

template<class TorchModule, class TorchDataset, class DatasetTransformation, class Sampler, class Optimizer, class Loss>
typename Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler, Optimizer, Loss>::TorchDataLoader Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler, Optimizer, Loss>::setup_data_loader(TorchDataset &dataset, int batchSize) {
  torch::data::datasets::MapDataset datasetMap = dataset.map(DatasetTransformation());

  torch::data::DataLoaderOptions dataLoaderOptions = torch::data::DataLoaderOptions().batch_size(batchSize);
  auto dataLoader = torch::data::make_data_loader<Sampler>(datasetMap, dataLoaderOptions);

  return dataLoader;
}
