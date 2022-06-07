//
// Created by Kartik Rajeshwaran on 2022-06-03.
//

#ifndef LIBTORCHPLAYGROUND_TRAINER_CPP_UTILITIES_METRICS_METRICS_H_
#define LIBTORCHPLAYGROUND_TRAINER_CPP_UTILITIES_METRICS_METRICS_H_

#include <torch/torch.h>

class Metrics {

 public:
  static float accuracy(torch::Tensor &predictedClasses, torch::Tensor &trueClasses);
  static std::vector<float> get_batch_metrics(std::vector<float> &losses, std::vector<float> &accuracies);
};

#endif//LIBTORCHPLAYGROUND_TRAINER_CPP_UTILITIES_METRICS_METRICS_H_
