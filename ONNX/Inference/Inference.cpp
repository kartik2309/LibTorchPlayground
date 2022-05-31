//
// Created by Kartik Rajeshwaran on 2022-05-30.
//

#include "Inference.h"

Inference::Inference(std::string &path, std::vector<std::string> &inputNames, std::vector<std::string> &outputNames) {

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
  session_ = new Ort::Session(env, path.c_str(), Ort::SessionOptions{nullptr});

  inputNames_ = inputNames;
  outputNames_ = outputNames;
}

torch::Tensor Inference::run_inference(torch::Tensor &input, float initFillValue, std::optional<int>numClasses) {
  std::vector<Ort::Value> inputTensors;
  std::vector<Ort::Value> outputTensors;
  int numClasses_ = numClasses.has_value() ? numClasses.value() : 10;

  std::vector<int64_t> inputSizes = get_tensor_size(input);
  assert(input.size(0) == 1);

  torch::TensorOptions outputTensorOptions = torch::TensorOptions().dtype(torch::kFloat);
  torch::Tensor output = torch::full({input.size(0), numClasses_}, initFillValue, outputTensorOptions);
  std::vector<int64_t> outputSizes = get_tensor_size(output);

  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
                                                         input.data_ptr<float>(),
                                                         input.numel(),
                                                         inputSizes.data(),
                                                         inputSizes.size()));
  outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
                                                        output.data_ptr<float>(),
                                                        output.numel(),
                                                        outputSizes.data(),
                                                        outputSizes.size()));

  auto inputNames = get_names(inputNames_);
  auto outputNames = get_names(outputNames_);

  try {
    session_->Run(Ort::RunOptions{nullptr},
                  inputNames.data(),
                  inputTensors.data(),
                  input.size(0),
                  outputNames.data(),
                  outputTensors.data(),
                  output.size(0));
  } catch (const Ort::Exception &exception) {
    BOOST_LOG_TRIVIAL(error) << "ERROR running model inference: " << exception.what() << std::endl;
    exit(-1);
  }

  return output;
}

std::vector<int64_t> Inference::get_tensor_size(torch::Tensor &tensor) {
  torch::IntArrayRef sizes = tensor.sizes();
  std::vector<int64_t> sizesVec;
  for (auto &size : sizes) {
    sizesVec.push_back(size);
  }
  return sizesVec;
}

std::vector<const char *> Inference::get_names(std::vector<std::string> &names) {
  std::vector<const char *> names_;
  for (auto &name : names) {
    names_.push_back(name.c_str());
  }
  return names_;
}

Inference::~Inference() = default;
