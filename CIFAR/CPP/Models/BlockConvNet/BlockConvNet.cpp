//
// Created by Kartik Rajeshwaran on 2022-05-19.
//

#include "BlockConvNet.h"

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
