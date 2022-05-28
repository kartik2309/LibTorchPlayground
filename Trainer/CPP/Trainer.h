//
// Created by Kartik Rajeshwaran on 2022-05-23.
//

#ifndef LIBTORCHPLAYGROUND_TRAINER_CPP_TRAINER_H_
#define LIBTORCHPLAYGROUND_TRAINER_CPP_TRAINER_H_
#define BOOST_LOG_DYN_LINK 1

#include "Utilities/BatchTransformTemplate/BatchTransformTemplate.h"
#include <boost/log/trivial.hpp>
#include <torch/torch.h>
#include <vector>
using namespace torch::indexing;

template<class TorchModule,
         class TorchDataset,
         class Sampler,
         class Optimizer,
         class Loss,
         class ReturnType = torch::Tensor>
class Trainer {

 protected:
  // Variables
  typedef torch::disable_if_t<
      torch::data::datasets::MapDataset<
          TorchDataset, BatchTransformTemplate<std::vector<torch::data::Example<>>, torch::data::Example<ReturnType, torch::Tensor>>>::is_stateful
          || !std::is_constructible<Sampler, size_t>::value,
      std::unique_ptr<torch::data::StatelessDataLoader<
          torch::data::datasets::MapDataset<
              TorchDataset,
              BatchTransformTemplate<std::vector<torch::data::Example<>>,
                                     torch::data::Example<ReturnType, torch::Tensor>>>,
          Sampler>>>
      TorchDataLoader;

  TorchModule model_;
  Optimizer *optimizer_;
  Loss *loss_;
  TorchDataLoader trainDataLoader_;
  TorchDataLoader evalDataLoader_;
  TorchDataLoader testDataLoader_;

  torch::DeviceType device_;

  // Functions
  std::vector<float> train_loop();
  std::vector<float> eval_loop();
  std::vector<float> test_loop();

  float accuracy(torch::Tensor &predictedClasses, torch::Tensor &trueClasses);
  std::vector<float> get_batch_metrics(std::vector<float> &losses, std::vector<float> &accuracies);

  TorchDataLoader setup_data_loader(TorchDataset &dataset, int batchSize);
  std::string handle_path(std::string &path);
  torch::Tensor to_tensor(torch::Tensor tensor);
  torch::Tensor to_tensor(torch::jit::IValue value);

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

  // Destructor
  ~Trainer();

  // Functions
  void fit(int epochs);
  void validate();
  void inference();
  void save_optimizer(std::string &path);
  void load_optimizer(std::string &path);
};

// ------------------------------------------ DEFINITIONS ------------------------------------------------------

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
std::vector<float> Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::train_loop() {
  std::vector<float> losses;
  std::vector<float> accuracies;

  model_->train();
  for (auto &batch : *trainDataLoader_) {

    auto data = batch.data;
    auto target = batch.target;

    torch::Tensor data_ = to_tensor(model_->forward(data));

    torch::Tensor predictedClasses = torch::argmax(data_, -1);

    optimizer_->zero_grad();
    torch::Tensor loss = loss_->ptr()->forward(data_, target);
    loss.backward();
    optimizer_->step();

    float accuracyValue = accuracy(predictedClasses, target);

    losses.push_back(loss.item<float>());
    accuracies.push_back(accuracyValue);
  }

  return get_batch_metrics(losses, accuracies);
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
std::vector<float> Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::eval_loop() {
  std::vector<float> losses;
  std::vector<float> accuracies;

  model_->eval();
  torch::NoGradGuard noGradGuard;
  {
    for (auto &batch : *evalDataLoader_) {
      auto data = batch.data;
      torch::Tensor target = batch.target;

      auto data_ = to_tensor(model_->forward(data));
      torch::Tensor predictedClasses = torch::argmax(data_, -1);

      torch::Tensor loss = loss_->ptr()->forward(data_, target);
      float accuracyValue = accuracy(predictedClasses, target);

      losses.push_back(loss.item<float>());
      accuracies.push_back(accuracyValue);
    }
  }

  return get_batch_metrics(losses, accuracies);
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
std::vector<float> Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::test_loop() {
  std::vector<float> losses;
  std::vector<float> accuracies;

  model_->eval();
  torch::NoGradGuard noGradGuard;
  {
    for (auto &batch : *testDataLoader_) {
      auto data = batch.data;
      torch::Tensor target = batch.target;

      auto data_ = to_tensor(model_->forward(data));
      torch::Tensor predictedClasses = torch::argmax(data_, -1);

      torch::Tensor loss = loss_->ptr()->forward(data_, target);
      float accuracyValue = accuracy(predictedClasses, target);

      losses.push_back(loss.item<float>());
      accuracies.push_back(accuracyValue);
    }
  }

  return get_batch_metrics(losses, accuracies);
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
float Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::accuracy(torch::Tensor &predictedClasses, torch::Tensor &trueClasses) {
  torch::Tensor ones_ = torch::ones(predictedClasses.sizes());
  torch::Tensor onesIndexed = ones_.index({predictedClasses.eq(trueClasses)});
  torch::Tensor mean_ = onesIndexed.sum() / ones_.sum();
  return mean_.item<float>();
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
std::vector<float> Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::get_batch_metrics(std::vector<float> &losses,
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
  float avgAccuracy = totalAccuracy / (float) accuracies.size();

  std::vector<float> metrics = {avgLoss, avgAccuracy};
  return metrics;
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
typename Trainer<TorchModule,
                 TorchDataset,
                 Sampler,
                 Optimizer,
                 Loss,
                 ReturnType>::TorchDataLoader
Trainer<TorchModule,
        TorchDataset,
        Sampler,
        Optimizer,
        Loss,
        ReturnType>::setup_data_loader(TorchDataset &dataset, int batchSize) {
  torch::data::datasets::MapDataset<TorchDataset, BatchTransformTemplate<std::vector<torch::data::Example<>>, torch::data::Example<ReturnType, torch::Tensor>>>

      datasetMap = dataset.map(BatchTransformTemplate<std::vector<torch::data::Example<>>, torch::data::Example<ReturnType, torch::Tensor>>());

  torch::data::DataLoaderOptions dataLoaderOptions = torch::data::DataLoaderOptions().batch_size(batchSize);
  auto dataLoader = torch::data::make_data_loader<Sampler>(datasetMap, dataLoaderOptions);

  return dataLoader;
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
std::string Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::handle_path(std::string &path) {
  std::filesystem::directory_entry de(path.c_str());
  if (de.is_directory()) {
    if (path.back() == '/') {
      path.append("Optimizer.pt");
      return path;
    }
    path.append("/Optimizer.pt");
    return path;
  }

  if (path.substr(path.length() - 3, path.length() - 1) != ".pt") {
    path = path.append(".pt");
  }
  return path;
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::Trainer(TorchModule &model,
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
  testDataLoader_ = nullptr;

  device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  model_->to(device_);
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::Trainer(TorchModule &model, TorchDataset &trainDataset, int batchSize, Optimizer &optimizer, Loss &loss) {

  model_ = model;
  optimizer_ = &optimizer;
  loss_ = &loss;
  trainDataLoader_ = setup_data_loader(trainDataset, batchSize);
  evalDataLoader_ = nullptr;
  testDataLoader_ = nullptr;

  device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  model_->to(device_);
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::Trainer(TorchModule &model, TorchDataset &trainDataset, TorchDataset &evalDataset, TorchDataset &testDataset, int batchSize, Optimizer &optimizer, Loss &loss) {
  model_ = model;
  optimizer_ = &optimizer;
  loss_ = &loss;
  trainDataLoader_ = setup_data_loader(trainDataset, batchSize);
  evalDataLoader_ = setup_data_loader(evalDataset);
  testDataLoader_ = setup_data_loader(testDataset);

  device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  model_->to(device_);
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::~Trainer() = default;

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
void Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::fit(int epochs) {

  for (int idx = 0; idx != epochs; idx++) {
    BOOST_LOG_TRIVIAL(info) << "Epoch:" << idx;
    BOOST_LOG_TRIVIAL(info) << "Training Phase";
    std::vector<float> trainMetrics = train_loop();
    BOOST_LOG_TRIVIAL(info) << "Loss:" << trainMetrics[0] << " Accuracy:" << trainMetrics[1];

    if (evalDataLoader_ != nullptr) {
      BOOST_LOG_TRIVIAL(info) << "Evaluation Phase";
      std::vector<float> evalMetrics = eval_loop();
      BOOST_LOG_TRIVIAL(info) << "Loss:" << evalMetrics[0] << " Accuracy:" << evalMetrics[1] << std::endl;
    }
  }
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
void Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::save_optimizer(std::string &path) {
  path = handle_path(path);
  torch::serialize::OutputArchive output_archive;
  optimizer_->save(output_archive);
  output_archive.save_to(path);
  BOOST_LOG_TRIVIAL(info) << ("Optimizer saved to " + path + "\n") << std::setw(0);
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
void Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::load_optimizer(std::string &path) {
  path = handle_path(path);
  torch::serialize::InputArchive archive;
  archive.load_from(path);
  optimizer_->load(archive);
  BOOST_LOG_TRIVIAL(info) << ("Optimizer loaded from " + path + "\n") << std::setw(0);
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
torch::Tensor Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::to_tensor(torch::Tensor tensor) {
  return tensor;
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
torch::Tensor Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::to_tensor(torch::jit::IValue value) {
  return value.toTensor();
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
void Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::inference() {
  if (testDataLoader_ == nullptr) {
    throw std::runtime_error("Expected to have Test DataLoader initialized. "
                             "Make sure you passed the Test Dataset in constructor.");
  }
  BOOST_LOG_TRIVIAL(info) << "Testing Phase";
  std::vector<float> evalMetrics = test_loop();
  BOOST_LOG_TRIVIAL(info) << "Loss:" << evalMetrics[0] << " Accuracy:" << evalMetrics[1] << std::endl;
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
void Trainer<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::validate() {
  if (evalDataLoader_ == nullptr) {
    throw std::runtime_error("Expected to have Evaluation DataLoader initialized. "
                             "Make sure you passed the Evaluation Dataset in constructor.");
  }

  BOOST_LOG_TRIVIAL(info) << "Validation Phase";
  std::vector<float> evalMetrics = eval_loop();
  BOOST_LOG_TRIVIAL(info) << "Loss:" << evalMetrics[0] << " Accuracy:" << evalMetrics[1] << std::endl;
}

#endif//LIBTORCHPLAYGROUND_TRAINER_CPP_TRAINER_H_
