//
// Created by Kartik Rajeshwaran on 2022-05-25.
//

#include "LoadJIT.h"

LoadJIT::LoadJIT(std::string &path) {
  model_ = torch::jit::load(path);
}
LoadJIT::LoadJIT(const char *path) {
  model_ = torch::jit::load(path);
}

LoadJIT::LoadJIT() = default;

LoadJIT::~LoadJIT() = default;

torch::jit::Module LoadJIT::get_model() {
  return model_;
}

torch::jit::Module LoadJIT::get_model(std::string &path) {
  model_ = torch::jit::load(path);
  return model_;
}

torch::jit::Module LoadJIT::get_model(const char *path) {
  model_ = torch::jit::load(path);
  return model_;
}

torch::jit::Module *LoadJIT::get_model_ptr() {
  return &model_;
}

torch::jit::Module *LoadJIT::get_model_ptr(std::string &path) {
  model_ = torch::jit::load(path);
  return &model_;
}

torch::jit::Module *LoadJIT::get_model_ptr(const char *path) {
  model_ = torch::jit::load(path);
  return &model_;
}
