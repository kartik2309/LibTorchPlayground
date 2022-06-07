//
// Created by Kartik Rajeshwaran on 2022-06-03.
//

#include "Metrics.h"

float Metrics::accuracy(torch::Tensor &predictedClasses, torch::Tensor &trueClasses) {
  torch::Tensor ones_ = torch::ones(predictedClasses.sizes());
  torch::Tensor onesIndexed = ones_.index({predictedClasses.eq(trueClasses)});
  torch::Tensor mean_ = onesIndexed.sum() / ones_.sum();
  return mean_.item<float>();
}

std::vector<float> Metrics::get_batch_metrics(std::vector<float> &losses,
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
