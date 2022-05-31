//
// Created by Kartik Rajeshwaran on 2022-05-30.
//

#ifndef LIBTORCHPLAYGROUND_ONNX_INFERENCE_INFERENCE_H_
#define LIBTORCHPLAYGROUND_ONNX_INFERENCE_INFERENCE_H_
#define BOOST_LOG_DYN_LINK 1

#include <torch/torch.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <boost/log/trivial.hpp>

class Inference {

  Ort::Session *session_;
  std::vector<std::string> inputNames_;
  std::vector<std::string> outputNames_;

  static std::vector<int64_t> get_tensor_size(torch::Tensor &tensor);
  static std::vector<const char *> get_names(std::vector<std::string> &names);

 public:
  Inference(std::string &path, std::vector<std::string> &inputNames, std::vector<std::string> &outputNames);
  ~Inference();

  torch::Tensor run_inference(torch::Tensor &input, float initFillValue = -100.23, std::optional<int>numClasses = std::optional<int>());
};

#endif//LIBTORCHPLAYGROUND_ONNX_INFERENCE_INFERENCE_H_
