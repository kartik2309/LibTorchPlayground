//
// Created by Kartik Rajeshwaran on 2022-05-19.
//

#include "BlockConvNet.h"

BlockConvNet::ModuleArgs::ModuleArgs(torch::Tensor &imageDims,
                                     torch::Tensor &channels,
                                     torch::Tensor &kernelSizesConv,
                                     torch::Tensor &stridesConv,
                                     torch::Tensor &dilationConv,
                                     torch::Tensor &kernelSizesPool,
                                     torch::Tensor &stridesPool,
                                     torch::Tensor &dilationPool,
                                     torch::Tensor &dropout,
                                     torch::Tensor &numClasses) {
  imageDims_ = imageDims;
  channels_ = channels;
  kernelSizesConv_ = kernelSizesConv;
  stridesConv_ = stridesConv;
  dilationConv_ = dilationConv;
  kernelSizesPool_ = kernelSizesPool;
  stridesPool_ = stridesPool;
  dilationPool_ = dilationPool;
  dropout_ = dropout;
  numClasses_ = numClasses;
}

BlockConvNet::ModuleArgs::ModuleArgs() = default;

std::vector<std::vector<int64_t>> BlockConvNet::get_interim(std::vector<int64_t> &imageDims,
                                                            std::vector<int64_t> &kernelSizesConv,
                                                            std::vector<int64_t> &stridesConv,
                                                            std::vector<int64_t> &dilationConv,
                                                            std::vector<int64_t> &kernelSizesPool,
                                                            std::vector<int64_t> &stridesPool,
                                                            std::vector<int64_t> &dilationPool) {
  size_t numBlocks = kernelSizesConv.size();
  std::vector<std::vector<int64_t>> interimSizes;
  interimSizes.push_back(imageDims);

  for (int64_t idx = 0; idx != numBlocks; idx++) {

    int64_t interimSizeConvHeight = floor((((interimSizes[idx][0] - dilationConv[idx] * (kernelSizesConv[idx] - 1) - 1) / stridesConv[idx])) + 1);
    int64_t interimSizeConvWidth = floor((((interimSizes[idx][1] - dilationConv[idx] * (kernelSizesConv[idx] - 1) - 1) / stridesConv[idx])) + 1);

    int64_t interimSizePoolHeight = floor((((interimSizeConvHeight - dilationPool[idx] * (kernelSizesPool[idx] - 1) - 1) / stridesPool[idx])) + 1);
    int64_t interimSizePoolWidth = floor((((interimSizeConvWidth - dilationPool[idx] * (kernelSizesPool[idx] - 1) - 1) / stridesPool[idx])) + 1);

    interimSizes.push_back({interimSizePoolHeight, interimSizePoolWidth});
  }

  return interimSizes;
}

std::string BlockConvNet::handle_path(std::string &path) {
  std::filesystem::directory_entry de(path.c_str());
  if (de.is_directory()) {
    if (path.back() == '/') {
      path.append("BlockConvNet.pt");
      return path;
    }
    path.append("/BlockConvNet.pt");
    return path;
  }

  if (path.substr(path.length() - 3, path.length() - 1) != ".pt") {
    path = path.append(".pt");
  }
  return path;
}

BlockConvNet::BlockConvNet(std::vector<int64_t> &imageDims,
                           std::vector<int64_t> &channels,
                           std::vector<int64_t> &kernelSizesConv,
                           std::vector<int64_t> &stridesConv,
                           std::vector<int64_t> &dilationConv,
                           std::vector<int64_t> &kernelSizesPool,
                           std::vector<int64_t> &stridesPool,
                           std::vector<int64_t> &dilationPool,
                           float_t dropout,
                           int64_t numClasses) {

  size_t numBlocks = channels.size() - 1;
  assert(numBlocks == kernelSizesConv.size());
  assert(kernelSizesConv.size() == stridesConv.size());
  assert(stridesConv.size() == dilationConv.size());
  assert(dilationConv.size() == kernelSizesPool.size());

  torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);

  torch::Tensor imageDims_ = torch::from_blob(imageDims.data(), {static_cast<long long>(imageDims.size())}, options);
  torch::Tensor channels_ = torch::from_blob(channels.data(), {static_cast<long long>(channels.size())}, options);
  torch::Tensor kernelSizesConv_ = torch::from_blob(kernelSizesConv.data(), {static_cast<long long>(kernelSizesConv.size())}, options);
  torch::Tensor stridesConv_ = torch::from_blob(stridesConv.data(), {static_cast<long long>(stridesConv.size())}, options);
  torch::Tensor dilationConv_ = torch::from_blob(dilationConv.data(), {static_cast<long long>(dilationConv.size())}, options);
  torch::Tensor kernelSizesPool_ = torch::from_blob(kernelSizesPool.data(), {static_cast<long long>(kernelSizesPool.size())}, options);
  torch::Tensor stridesPool_ = torch::from_blob(stridesPool.data(), {static_cast<long long>(stridesPool.size())}, options);
  torch::Tensor dilationPool_ = torch::from_blob(dilationPool.data(), {static_cast<long long>(dilationPool.size())}, options);
  torch::Tensor dropout_ = torch::full({}, dropout, options);
  torch::Tensor numClasses_ = torch::full({}, numClasses, options);

  moduleArgs = ModuleArgs(imageDims_,
                          channels_,
                          kernelSizesConv_,
                          stridesConv_,
                          dilationConv_,
                          kernelSizesPool_,
                          stridesPool_,
                          dilationPool_,
                          dropout_,
                          numClasses_);

  setup_model(
      imageDims,
      channels,
      kernelSizesConv,
      stridesConv,
      dilationConv,
      kernelSizesPool,
      stridesPool,
      dilationPool,
      dropout,
      numClasses);
}

BlockConvNet::~BlockConvNet() = default;

torch::Tensor BlockConvNet::forward(torch::Tensor x) {
  assert(convSubmodules->size() == maxPoolSubmodules->size());
  for (int64_t idx = 0; idx != convSubmodules->size(); idx++) {
    x = convSubmodules[idx]->as<torch::nn::Conv2d>()->forward(x);
    x = relu(x);
    x = maxPoolSubmodules[idx]->as<torch::nn::MaxPool2d>()->forward(x);
  }
  x = flattenSubmodule->ptr()->forward(x);
  x = dropoutSubmodule->ptr()->forward(x);
  x = linearSubmodule->ptr()->forward(x);
  return x;
}

void BlockConvNet::save_model(std::string &path) {
  path = handle_path(path);
  torch::serialize::OutputArchive output_archive;
  save(output_archive);

  output_archive.write("imageDims", moduleArgs.imageDims_);
  output_archive.write("channels", moduleArgs.channels_);
  output_archive.write("kernelSizesConv", moduleArgs.kernelSizesConv_);
  output_archive.write("stridesConv", moduleArgs.stridesConv_);
  output_archive.write("dilationConv", moduleArgs.dilationConv_);
  output_archive.write("kernelSizesPool", moduleArgs.kernelSizesPool_);
  output_archive.write("stridesConv", moduleArgs.stridesConv_);
  output_archive.write("dilationPool", moduleArgs.dilationPool_);
  output_archive.write("dropoutVal", moduleArgs.dropout_);
  output_archive.write("numClasses", moduleArgs.numClasses_);

  output_archive.save_to(path);

  BOOST_LOG_TRIVIAL(info) << ("Model saved to " + path + "\n") << std::setw(0);
}

void BlockConvNet::load_model(std::string &path) {
  path = handle_path(path);
  torch::serialize::InputArchive archive;
  archive.load_from(path);
  load(archive);
  BOOST_LOG_TRIVIAL(info) << "Model loaded from " << path << "\n";
}


BlockConvNet::BlockConvNet(std::string &path, std::optional<float_t> dropout_) {
  path = handle_path(path);
  torch::serialize::InputArchive archive;
  archive.load_from(path);

  archive.read("imageDims", moduleArgs.imageDims_);
  archive.read("channels", moduleArgs.channels_);
  archive.read("kernelSizesConv", moduleArgs.kernelSizesConv_);
  archive.read("stridesConv", moduleArgs.stridesConv_);
  archive.read("dilationConv", moduleArgs.dilationConv_);
  archive.read("kernelSizesPool", moduleArgs.kernelSizesPool_);
  archive.read("stridesConv", moduleArgs.stridesConv_);
  archive.read("dilationPool", moduleArgs.dilationPool_);
  archive.read("dropoutVal", moduleArgs.dropout_);
  archive.read("numClasses", moduleArgs.numClasses_);

  std::vector<int64_t> imageDims = inverse_stack(moduleArgs.imageDims_);
  std::vector<int64_t> channels = inverse_stack(moduleArgs.channels_);
  std::vector<int64_t> kernelSizesConv = inverse_stack(moduleArgs.kernelSizesConv_);
  std::vector<int64_t> stridesConv = inverse_stack(moduleArgs.stridesConv_);
  std::vector<int64_t> dilationConv = inverse_stack(moduleArgs.dilationConv_);
  std::vector<int64_t> kernelSizesPool = inverse_stack(moduleArgs.kernelSizesPool_);
  std::vector<int64_t> stridesPool = inverse_stack(moduleArgs.stridesConv_);
  std::vector<int64_t> dilationPool = inverse_stack(moduleArgs.dilationPool_);
  float_t dropout = !dropout_.has_value() ? moduleArgs.dropout_.item<float_t>() : dropout_.value();
  int64_t numClasses = moduleArgs.numClasses_.item<int64_t>();

  setup_model(imageDims,
              channels,
              kernelSizesConv,
              stridesConv,
              dilationConv,
              kernelSizesPool,
              stridesPool,
              dilationPool,
              dropout,
              numClasses);
}

void BlockConvNet::setup_model(std::vector<int64_t> &imageDims,
                               std::vector<int64_t> &channels,
                               std::vector<int64_t> &kernelSizesConv,
                               std::vector<int64_t> &stridesConv,
                               std::vector<int64_t> &dilationConv,
                               std::vector<int64_t> &kernelSizesPool,
                               std::vector<int64_t> &stridesPool,
                               std::vector<int64_t> &dilationPool,
                               float_t dropout,
                               int64_t numClasses) {

  size_t numBlocks = channels.size() - 1;

  std::string convModuleName = "conv_";
  std::string maxPoolModuleName = "pool_";

  for (int64_t idx = 0; idx != numBlocks; idx++) {
    torch::nn::Conv2dOptions conv2dOptions = torch::nn::Conv2dOptions(channels[idx],
                                                                      channels[idx + 1],
                                                                      kernelSizesConv[idx])
                                                 .stride(stridesConv[idx])
                                                 .dilation(dilationConv[idx]);
    auto *convBlock = new torch::nn::Conv2d(conv2dOptions);

    auto maxPool2dOptions = torch::nn::MaxPool2dOptions(
                                kernelSizesConv[idx])
                                .dilation(dilationPool[idx])
                                .stride(stridesPool[idx]);

    auto *maxPoolBlock = new torch::nn::MaxPool2d(maxPool2dOptions);

    convSubmodules->push_back(*convBlock);
    maxPoolSubmodules->push_back(*maxPoolBlock);

    register_module(convModuleName.append((std::to_string(idx))), *convBlock);
    register_module(maxPoolModuleName.append((std::to_string(idx))), *maxPoolBlock);
  }
  std::vector<std::vector<int64_t>> interimSizes = get_interim(imageDims,
                                                               kernelSizesConv,
                                                               stridesConv,
                                                               dilationConv,
                                                               kernelSizesPool,
                                                               stridesPool,
                                                               dilationPool);
  std::vector<int64_t> finalSizes = interimSizes.back();

  auto dropoutOptions = torch::nn::DropoutOptions(dropout);
  dropoutSubmodule = new torch::nn::Dropout(dropoutOptions);
  register_module("dropout", *dropoutSubmodule);

  auto flattenOptions = torch::nn::FlattenOptions().start_dim(1).end_dim(-1);
  flattenSubmodule = new torch::nn::Flatten(flattenOptions);
  register_module("flatten", *flattenSubmodule);

  auto linearOptions = torch::nn::LinearOptions(finalSizes[0] * finalSizes[1] * channels.back(), numClasses);
  linearSubmodule = new torch::nn::Linear(linearOptions);
  register_module("linear", *linearSubmodule);
}
std::vector<int64_t> BlockConvNet::inverse_stack(torch::Tensor &tensor) {
  int64_t size = tensor.size(0);
  std::vector<int64_t> tensorVector;

  for (int idx = 0; idx != size; idx++) {
    tensorVector.push_back(tensor[idx].item<int64_t>());
  }

  return tensorVector;
}
