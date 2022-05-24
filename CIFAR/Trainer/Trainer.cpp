//
// Created by Kartik Rajeshwaran on 2022-05-23.
//

#include "Trainer.h"

template<class TorchModule, class TorchDataset, class DatasetTransformation, class Sampler>
std::vector<float> Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler>::train_loop(torch::optim::AdamW &optimizer) {
  std::vector<float> losses;
  std::vector<float> accuracies;
  int num_batches = 0;

  model_->train();
  for (torch::data::Example<> &batch : *trainDataLoader_) {
    torch::Tensor data = batch.data;
    torch::Tensor target = batch.target;

    data = model_->forward(data);
    torch::Tensor predictedClasses = torch::argmax(data, -1);

    optimizer.zero_grad();
    torch::Tensor loss = crossEntropyLoss->ptr()->forward(data, target);
    loss.backward();
    optimizer.step();

    float accuracyValue = accuracy(predictedClasses, target);

    losses.push_back(loss.item<float>());
    accuracies.push_back(accuracyValue);

    num_batches += 1;
  }

  return get_batch_metrics(losses, accuracies, num_batches);
}

template<class TorchModule, class TorchDataset, class DatasetTransformation, class Sampler>
std::vector<float> Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler>::eval_loop() {
  std::vector<float> losses;
  std::vector<float> accuracies;
  int num_batches = 0;

  model_->eval();
  for (torch::data::Example<> &batch : *evalDataLoader_) {
    torch::Tensor data = batch.data;
    torch::Tensor target = batch.target;

    data = model_->forward(data);
    torch::Tensor predictedClasses = torch::argmax(data, -1);

    torch::Tensor loss = crossEntropyLoss->ptr()->forward(data, target);
    float accuracyValue = accuracy(predictedClasses, target);

    losses.push_back(loss.item<float>());
    accuracies.push_back(accuracyValue);
    num_batches += 1;
  }

  return get_batch_metrics(losses, accuracies, num_batches);
}

template<class TorchModule, class TorchDataset, class DatasetTransformation, class Sampler>
float Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler>::accuracy(torch::Tensor &predictedClasses, torch::Tensor &trueClasses) {
  torch::Tensor ones_ = torch::ones(predictedClasses.sizes());
  torch::Tensor onesIndexed = ones_.index({predictedClasses.eq(trueClasses)});
  torch::Tensor mean_ = onesIndexed.sum() / ones_.sum();
  return mean_.item<float>();
}

template<class TorchModule, class TorchDataset, class DatasetTransformation, class Sampler>
std::vector<float> Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler>::get_batch_metrics(std::vector<float> &losses,
                                                                                                         std::vector<float> &accuracies,
                                                                                                         int num_batches) {
  float totalLoss = 0.00;
  for (auto &value : losses) {
    totalLoss += value;
  }
  float avgLoss = totalLoss / (float) num_batches;

  float totalAccuracy = 0.00;
  for (auto &value : accuracies) {
    totalAccuracy += value;
  }
  float avgAccuracy = totalAccuracy / (float) num_batches;

  std::vector<float> metrics = {avgLoss, avgAccuracy};
  return metrics;
}

template<class TorchModule, class TorchDataset, class DatasetTransformation, class Sampler>
Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler>::Trainer(TorchModule &model,
                                                                            TorchDataset &trainDataset,
                                                                            TorchDataset &evalDataset,
                                                                            int batchSize,
                                                                            float lr) {

  model_ = model;
  lr_ = lr;
  trainDataLoader_ = setup_data_loader(trainDataset, batchSize);
  evalDataLoader_ = setup_data_loader(evalDataset, batchSize);

  torch::nn::CrossEntropyLossOptions crossEntropyLossOptions = torch::nn::CrossEntropyLossOptions().reduction(torch::kMean);
  crossEntropyLoss = new torch::nn::CrossEntropyLoss(crossEntropyLossOptions);
}

template<class TorchModule, class TorchDataset, class DatasetTransformation, class Sampler>
Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler>::~Trainer() = default;

template<class TorchModule, class TorchDataset, class DatasetTransformation, class Sampler>
void Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler>::fit(int epochs) {

  auto adamWOptions = torch::optim::AdamWOptions(lr_);
  torch::optim::AdamW adamOptimizer(model_->parameters(), adamWOptions);

  for (int idx = 0; idx != epochs; idx++) {
    std::cout << "Epoch:" << idx << std::endl;
    std::cout << "Training Phase" << std::endl;
    std::vector<float> trainMetrics = train_loop(adamOptimizer);
    std::cout << "Loss:" << trainMetrics[0] << " Accuracy:" << trainMetrics[1] << std::endl;

    std::cout << "Evaluation Phase" << std::endl;
    std::vector<float> evalMetrics = eval_loop();
    std::cout << "Loss:" << evalMetrics[0] << " Accuracy:" << evalMetrics[1] << std::endl;
    std::cout << std::endl;
  }
}

template<class TorchModule, class TorchDataset, class DatasetTransformation, class Sampler>
typename Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler>::TorchDataLoader Trainer<TorchModule, TorchDataset, DatasetTransformation, Sampler>::setup_data_loader(TorchDataset &dataset, int batchSize) {
  torch::data::datasets::MapDataset datasetMap = dataset.map(DatasetTransformation());

  torch::data::DataLoaderOptions dataLoaderOptions = torch::data::DataLoaderOptions().batch_size(batchSize);
  auto dataLoader = torch::data::make_data_loader<Sampler>(datasetMap, dataLoaderOptions);

  return dataLoader;
}
