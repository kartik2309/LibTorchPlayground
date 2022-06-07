//
// Created by Kartik Rajeshwaran on 2022-06-04.
//

#ifndef LIBTORCHPLAYGROUND_TRAINER_CPP_TRAINERBASE_HPP_
#define LIBTORCHPLAYGROUND_TRAINER_CPP_TRAINERBASE_HPP_
#define BOOST_LOG_DYN_LINK 1

#include <boost/log/trivial.hpp>
#include <torch/torch.h>
#include <vector>

#include "Utilities/BatchTransformTemplate.hpp"
#include "Utilities/MultiprocessingBackend.hpp"
#include "Utilities/Metrics/Metrics.h"

using namespace torch::indexing;

template<class TorchModule,
         class TorchDataset,
         class Sampler,
         class Optimizer,
         class Loss,
         class ReturnType = torch::Tensor>
class TrainerBase {

 protected:
  // Variables
  typedef torch::disable_if_t<
      torch::data::datasets::MapDataset<
          TorchDataset,
          BatchTransformTemplate<std::vector<torch::data::Example<>>,
                                 torch::data::Example<ReturnType, torch::Tensor>>>::is_stateful
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

  int n_procs_ = -1;

  MultiprocessingBackend<TorchModule, TorchDataset, Optimizer, Loss> *mpb = nullptr;

  torch::DeviceType device_;

  // Functions
  virtual void train_loop();
  virtual void eval_loop();
  virtual void test_loop();

  torch::Tensor train_step(torch::Tensor &logits, torch::Tensor &targets);
  torch::Tensor eval_step(torch::Tensor &logits, torch::Tensor &targets);

  TorchDataLoader setup_data_loader(TorchDataset &dataset, int batchSize);
  std::string handle_path(std::string &path);
  torch::Tensor to_tensor(torch::Tensor tensor);
  torch::Tensor to_tensor(torch::jit::IValue value);

 public:
  // Constructor
  TrainerBase(TorchModule &model,
          TorchDataset &trainDataset,
          TorchDataset &evalDataset,
          int batchSize,
          Optimizer &optimizer,
          Loss &loss);

  TrainerBase(TorchModule &model,
          TorchDataset &trainDataset,
          int batchSize,
          Optimizer &optimizer,
          Loss &loss);

  TrainerBase(TorchModule &model,
          TorchDataset &trainDataset,
          TorchDataset &evalDataset,
          TorchDataset &testDataset,
          int batchSize,
          Optimizer &optimizer,
          Loss &loss);

  TrainerBase(TorchModule &model,
          TorchDataset &trainDataset,
          TorchDataset &evalDataset,
          int batchSize,
          Optimizer &optimizer,
          Loss &loss,
          int n_procs);

  // Destructor
  ~TrainerBase();

  // Functions
  virtual void fit(int epochs);
  virtual void fit_parallel(int epochs, std::optional<int> log_from_rank = std::optional<int>());
  virtual void validate();
  virtual void inference();
  void save_optimizer(std::string &path);
  void load_optimizer(std::string &path);
};

// ------------------------------------------ DEFINITIONS ------------------------------------------------------

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
void TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::train_loop() {

  model_->train();
  for (auto &batch : *trainDataLoader_) {

    auto data = batch.data;
    auto target = batch.target;

    torch::Tensor logits = to_tensor(model_->forward(data));
    train_step(logits, target);
  }
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
void TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::eval_loop() {

  model_->eval();
  torch::NoGradGuard noGradGuard;
  {
    for (auto &batch : *evalDataLoader_) {
      auto data = batch.data;
      torch::Tensor target = batch.target;
      auto data_ = to_tensor(model_->forward(data));
      torch::Tensor loss = loss_->ptr()->forward(data_, target);
    }
  }
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
void TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::test_loop() {
  
  model_->eval();
  torch::NoGradGuard noGradGuard;
  {
    for (auto &batch : *testDataLoader_) {
      auto data = batch.data;
      torch::Tensor target = batch.target;

      auto data_ = to_tensor(model_->forward(data));

      torch::Tensor loss = loss_->ptr()->forward(data_, target);
    }
  }
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
torch::Tensor TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::train_step(torch::Tensor &logits, torch::Tensor &targets) {
  optimizer_->zero_grad();
  torch::Tensor loss = loss_->ptr()->forward(logits, targets);
  loss.backward();
  optimizer_->step();

  return loss;
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
torch::Tensor TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::eval_step(torch::Tensor &logits, torch::Tensor &targets) {
  torch::Tensor loss = loss_->ptr()->forward(logits, targets);
  return loss;
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
typename TrainerBase<TorchModule,
                 TorchDataset,
                 Sampler,
                 Optimizer,
                 Loss,
                 ReturnType>::TorchDataLoader
TrainerBase<TorchModule,
        TorchDataset,
        Sampler,
        Optimizer,
        Loss,
        ReturnType>::setup_data_loader(TorchDataset &dataset, int batchSize) {

  torch::data::datasets::MapDataset<TorchDataset,
                                    BatchTransformTemplate<std::vector<torch::data::Example<>>,
                                                           torch::data::Example<ReturnType,
                                                                                torch::Tensor>>>
      datasetMap = dataset.map(BatchTransformTemplate<std::vector<torch::data::Example<>>,
                                                      torch::data::Example<ReturnType, torch::Tensor>>());

  torch::data::DataLoaderOptions dataLoaderOptions = torch::data::DataLoaderOptions().batch_size(batchSize);
  auto dataLoader = torch::data::make_data_loader<Sampler>(datasetMap, dataLoaderOptions);

  return dataLoader;
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
std::string TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::handle_path(std::string &path) {
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
TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::TrainerBase(TorchModule &model,
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
TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::TrainerBase(TorchModule &model, TorchDataset &trainDataset, int batchSize, Optimizer &optimizer, Loss &loss) {

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
TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::TrainerBase(TorchModule &model, TorchDataset &trainDataset, TorchDataset &evalDataset, TorchDataset &testDataset, int batchSize, Optimizer &optimizer, Loss &loss) {
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
TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::TrainerBase(TorchModule &model,
                                                                                  TorchDataset &trainDataset,
                                                                                  TorchDataset &evalDataset,
                                                                                  int batchSize,
                                                                                  Optimizer &optimizer,
                                                                                  Loss &loss,
                                                                                  int n_procs) {
  if (n_procs > 1){
    mpb = new MultiprocessingBackend<TorchModule, TorchDataset, Optimizer, Loss>(
        model,
        trainDataset,
        batchSize,
        optimizer,
        loss);

    device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    n_procs_ = n_procs;
  }
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::~TrainerBase() = default;

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
void TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::fit(int epochs) {

  if (n_procs_ > -1){
    throw std::runtime_error("Initialized with n_procs argument in constructor. Use fit_parallel method to train!");
  }

  for (int epoch = 0; epoch != epochs; epoch++) {
    BOOST_LOG_TRIVIAL(info) << "Epoch:" << epoch;
    BOOST_LOG_TRIVIAL(info) << "Training Phase";
    train_loop();

    if (evalDataLoader_ != nullptr) {
      BOOST_LOG_TRIVIAL(info) << "Evaluation Phase";
      eval_loop();
    }
  }
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
void TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::fit_parallel(int epochs, std::optional<int> log_from_rank) {
  if (n_procs_ == -1){
    throw std::runtime_error("Initialized without n_procs argument in constructor. Use fit method to train!");
  }
  int log_from_rank_ = log_from_rank.has_value() ? log_from_rank.value() : -1;
  mpb->fit(epochs, log_from_rank_);
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
void TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::save_optimizer(std::string &path) {
  path = handle_path(path);
  torch::serialize::OutputArchive output_archive;
  optimizer_->save(output_archive);
  output_archive.save_to(path);
  BOOST_LOG_TRIVIAL(info) << ("Optimizer saved to " + path + "\n") << std::setw(0);
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
void TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::load_optimizer(std::string &path) {
  path = handle_path(path);
  torch::serialize::InputArchive archive;
  archive.load_from(path);
  optimizer_->load(archive);
  BOOST_LOG_TRIVIAL(info) << ("Optimizer loaded from " + path + "\n") << std::setw(0);
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
torch::Tensor TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::to_tensor(torch::Tensor tensor) {
  return tensor;
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
torch::Tensor TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::to_tensor(torch::jit::IValue value) {
  return value.toTensor();
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
void TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::inference() {
  if (testDataLoader_ == nullptr) {
    throw std::runtime_error("Expected to have Test DataLoader initialized. "
                             "Make sure you passed the Test Dataset in constructor.");
  }
  BOOST_LOG_TRIVIAL(info) << "Testing Phase";
}

template<class TorchModule, class TorchDataset, class Sampler, class Optimizer, class Loss, class ReturnType>
void TrainerBase<TorchModule, TorchDataset, Sampler, Optimizer, Loss, ReturnType>::validate() {
  if (evalDataLoader_ == nullptr) {
    throw std::runtime_error("Expected to have Evaluation DataLoader initialized. "
                             "Make sure you passed the Evaluation Dataset in constructor.");
  }

  BOOST_LOG_TRIVIAL(info) << "Validation Phase";
  eval_loop();
}

#endif//LIBTORCHPLAYGROUND_TRAINER_CPP_TRAINERBASE_HPP_
