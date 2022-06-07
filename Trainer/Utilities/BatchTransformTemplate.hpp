//
// Created by Kartik Rajeshwaran on 2022-05-25.
//

#ifndef LIBTORCHPLAYGROUND_TRAINER_CPP_UTILITIES_BATCHTRANSFORMTEMPLATE_BATCHTRANSFORMTEMPLATE_HPP_
#define LIBTORCHPLAYGROUND_TRAINER_CPP_UTILITIES_BATCHTRANSFORMTEMPLATE_BATCHTRANSFORMTEMPLATE_HPP_

#include <torch/torch.h>

template<typename InputType = std::vector<torch::data::Example<>>,
         typename OutputType = torch::data::Example<>>
class BatchTransformTemplate : public torch::data::transforms::BatchTransform<InputType, OutputType> {

 protected:
  c10::Device *device_;

  // Functions
  void set_data(torch::Tensor &tensors, std::vector<torch::jit::IValue> &outData);
  void set_data(torch::Tensor &tensors, torch::Tensor &outData);
  OutputType apply_batch_(InputType examples);

 public:
  // Types
  using typename torch::data::transforms::BatchTransform<InputType, OutputType>::InputBatchType;
  using typename torch::data::transforms::BatchTransform<InputType, OutputType>::OutputBatchType;

  // Constructor
  explicit BatchTransformTemplate(std::optional<c10::DeviceIndex> deviceIndex = std::optional<c10::DeviceIndex>());

  // Destructor
  ~BatchTransformTemplate();

  // Functions
  OutputType apply_batch(InputType examples) override;
};

// ------------------------------------------ DEFINITIONS ------------------------------------------------------ //

template<typename InputType, typename OutputType>
OutputType BatchTransformTemplate<InputType, OutputType>::apply_batch_(InputType examples) {

  OutputType outputExample;
  std::vector<torch::Tensor> outputData;
  std::vector<torch::Tensor> outputTarget;

  outputData.reserve(examples.size());
  outputTarget.reserve(examples.size());

  for (auto &example : examples) {
    outputData.push_back(std::move(example.data.to(device_->str())));
    outputTarget.push_back(std::move(example.target).to(device_->str()));
  }

  torch::Tensor stackedTensorsData = torch::stack(outputData);
  torch::Tensor stackedTensorsTarget = torch::stack(outputTarget);

  set_data(stackedTensorsData, outputExample.data);
  outputExample.target = stackedTensorsTarget;

  return outputExample;
}

template<typename InputType, typename OutputType>
BatchTransformTemplate<InputType, OutputType>::BatchTransformTemplate(std::optional<c10::DeviceIndex> deviceIndex) {
  static_assert((std::is_same<typename OutputType::DataType, std::vector<torch::jit::IValue>>::value)
                or (std::is_same<typename OutputType::DataType, torch::Tensor>::value),
                "OutputType must be one of std::vector<torch::jit::IValue> or torch::Tensor");

  device_ = new torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  if (device_->is_cuda()){
    auto deviceIndex_ = deviceIndex.has_value() ? deviceIndex.value() : c10::DeviceIndex(0);
    device_->set_index(deviceIndex_);
  }
}

template<typename InputType, typename OutputType>
BatchTransformTemplate<InputType, OutputType>::~BatchTransformTemplate() = default;

template<typename InputType, typename OutputType>
OutputType BatchTransformTemplate<InputType, OutputType>::apply_batch(InputType examples) {
  return apply_batch_(examples);
}

template<typename InputType, typename OutputType>
void BatchTransformTemplate<InputType, OutputType>::set_data(torch::Tensor &tensors, std::vector<torch::jit::IValue> &outData) {
  std::vector<torch::jit::IValue> finalVector;
  finalVector.emplace_back(tensors);
  outData = finalVector;
}

template<typename InputType, typename OutputType>
void BatchTransformTemplate<InputType, OutputType>::set_data(torch::Tensor &tensors, torch::Tensor &outData) {
  outData = tensors;
}

#endif//LIBTORCHPLAYGROUND_TRAINER_CPP_UTILITIES_BATCHTRANSFORMTEMPLATE_BATCHTRANSFORMTEMPLATE_HPP_
