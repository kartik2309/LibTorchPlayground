//
// Created by Kartik Rajeshwaran on 2022-06-01.
//

#ifndef LIBTORCHPLAYGROUND_TRAINER_CPP_UTILITIES_MULTIPROCESSINGBACKEND_HPP_
#define LIBTORCHPLAYGROUND_TRAINER_CPP_UTILITIES_MULTIPROCESSINGBACKEND_HPP_
#define BOOST_LOG_DYN_LINK 1
#define MASTER 0

#include <boost/log/trivial.hpp>
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/vector.hpp>

#include "BatchTransformTemplate.hpp"
#include "Metrics/Metrics.h"
#include <torch/torch.h>

template<class TorchModule, class TorchDataset, class Optimizer, class Loss>
class MultiprocessingBackend {

  TorchModule model_;
  Optimizer *optimizer_;
  Loss *loss_;
  TorchDataset *trainDataset_;
  int batchSize_;
  int n_procs_;

  Metrics *metrics = new Metrics();

  torch::Tensor vector_to_tensor(std::vector<std::vector<float>> &vector);
  std::vector<std::vector<float>> tensor_to_vector(torch::Tensor &tensor);

 public:
  MultiprocessingBackend(TorchModule &model,
                         TorchDataset &trainDataset,
                         int batchSize,
                         Optimizer &optimizer,
                         Loss &loss,
                         int n_procs = 4);
  ~MultiprocessingBackend();

  void fit(int epochs, int log_from_rank = -1);
};

template<class TorchModule, class TorchDataset, class Optimizer, class Loss>
MultiprocessingBackend<TorchModule, TorchDataset, Optimizer, Loss>::MultiprocessingBackend(TorchModule &model,
                                                                                           TorchDataset &trainDataset,
                                                                                           int batchSize,
                                                                                           Optimizer &optimizer,
                                                                                           Loss &loss,
                                                                                           int n_procs) {

  model_ = model;
  trainDataset_ = &trainDataset;
  batchSize_ = batchSize;
  optimizer_ = &optimizer;
  loss_ = &loss;
  n_procs_ = n_procs;

  size_t deviceCount = torch::cuda::device_count();
  if (deviceCount > 0) {

    for (size_t deviceId; deviceId != deviceCount; deviceId++) {
      const std::string deviceIdString = "cuda:" + std::to_string(deviceId);
      model_->copy().to(deviceIdString);
    }
  }
}

template<class TorchModule, class TorchDataset, class Optimizer, class Loss>
std::vector<std::vector<float>> MultiprocessingBackend<TorchModule, TorchDataset, Optimizer, Loss>::tensor_to_vector(torch::Tensor &tensor) {
  auto flattenedTensor = tensor.flatten();

  std::vector<float> vector_(flattenedTensor.data_ptr<float>(),
                             flattenedTensor.data_ptr<float>() + flattenedTensor.numel());
  std::vector<float> shape_(tensor.sizes().begin(), tensor.sizes().end());
  std::vector<std::vector<float>> vector = {vector_, shape_};

  return vector;
}

template<class TorchModule, class TorchDataset, class Optimizer, class Loss>
void MultiprocessingBackend<TorchModule, TorchDataset, Optimizer, Loss>::fit(int epochs, int log_from_rank) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  if (log_from_rank > n_procs_) {
    log_from_rank = MASTER;
  }

  {
    auto datasetMap = trainDataset_->map(BatchTransformTemplate<std::vector<torch::data::Example<>>,
                                                                torch::data::Example<std::vector<torch::jit::IValue>,
                                                                                     torch::Tensor>>());
    torch::data::samplers::DistributedRandomSampler sampler(trainDataset_->size().value(), n_procs_, world.rank());

    auto dataLoaderOptions = torch::data::DataLoaderOptions().batch_size(batchSize_);
    auto dataLoader = torch::data::make_data_loader(datasetMap, sampler, dataLoaderOptions);

    std::vector<float> losses;
    std::vector<float> accuracies;

    model_->train();
    for (int epoch; epoch != epochs; epoch++) {

      if (log_from_rank > -1 and log_from_rank == world.rank()) {
        BOOST_LOG_TRIVIAL(info) << "EPOCH:" << epoch << std::endl;
      } else if (log_from_rank == -1) {
        BOOST_LOG_TRIVIAL(info) << "Rank:" << world.rank() << " EPOCH:" << epoch << std::endl;
      }

      sampler.reset();
      losses.clear();
      accuracies.clear();

      for (auto &batch : *dataLoader) {

        auto data = batch.data;
        auto target = batch.target;

        torch::Tensor data_ = model_->forward(data).toTensor();

        torch::Tensor predictedClasses = torch::argmax(data_, -1);

        optimizer_->zero_grad();
        torch::Tensor loss = loss_->ptr()->forward(data_, target);
        loss.backward();

        losses.push_back(loss.item<float>());
        accuracies.push_back(Metrics::accuracy(predictedClasses, target));

        world.barrier();

        for (auto param : model_->parameters()) {
          auto vectorGradient = tensor_to_vector(param.mutable_grad());
          std::vector<std::vector<float>> meanVectorGradient;

          if (world.rank() == MASTER) {
            std::vector<std::vector<std::vector<float>>> toGatherVector;
            std::vector<torch::Tensor> gatheredTensors;

            torch::Tensor tensor_;
            boost::mpi::gather(world, vectorGradient, toGatherVector, MASTER);

            for (auto &vec : toGatherVector) {
              tensor_ = vector_to_tensor(vec);
              gatheredTensors.push_back(tensor_);
            }

            torch::Tensor stacked = torch::stack(gatheredTensors);
            torch::Tensor meanTensor = torch::mean(stacked, 0);

            torch::Tensor flattened = meanTensor.flatten();
            std::vector<float> meanVectorGradient_(flattened.data_ptr<float>(),
                                                   flattened.data_ptr<float>() + flattened.numel());
            meanVectorGradient.push_back(meanVectorGradient_);

            std::vector<float> shapes(meanTensor.sizes().begin(), meanTensor.sizes().end());
            meanVectorGradient.push_back(shapes);
          } else {
            boost::mpi::gather(world, vectorGradient, MASTER);
          }

          boost::mpi::broadcast(world, meanVectorGradient, MASTER);
          torch::Tensor meanGradient = vector_to_tensor(meanVectorGradient);

          param.mutable_grad() = meanGradient;
          assert(meanGradient.equal(param.grad()));
        }
        world.barrier();

        optimizer_->step();
      }
      std::vector<float> trainMetrics = Metrics::get_batch_metrics(losses, accuracies);
      if (log_from_rank > -1 and world.rank() == log_from_rank) {
        BOOST_LOG_TRIVIAL(info) << "Loss:" << trainMetrics[0] << " Accuracy:" << trainMetrics[1] << std::endl;
      } else if (log_from_rank == -1) {
        BOOST_LOG_TRIVIAL(info) << "Rank:" << world.rank() << " Loss:" << trainMetrics[0] << " Accuracy:" << trainMetrics[1] << std::endl;
      }
    }
  }
}

template<class TorchModule, class TorchDataset, class Optimizer, class Loss>
torch::Tensor MultiprocessingBackend<TorchModule, TorchDataset, Optimizer, Loss>::vector_to_tensor(std::vector<std::vector<float>> &vector) {
  std::vector<float> vector_ = vector[0];
  std::vector<float> shape_ = vector[1];

  torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor tensor = torch::zeros(std::vector<int64_t>(shape_.begin(), shape_.end()), options);
  std::memmove(tensor.data_ptr<float>(), vector_.data(), sizeof(float) * tensor.numel());
  return tensor;
}

template<class TorchModule, class TorchDataset, class Optimizer, class Loss>
MultiprocessingBackend<TorchModule, TorchDataset, Optimizer, Loss>::~MultiprocessingBackend() = default;

#endif//LIBTORCHPLAYGROUND_TRAINER_CPP_UTILITIES_MULTIPROCESSINGBACKEND_HPP_