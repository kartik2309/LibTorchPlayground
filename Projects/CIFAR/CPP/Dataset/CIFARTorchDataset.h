//
// Created by Kartik Rajeshwaran on 2022-05-19.
//

#ifndef LIBTORCHPLAYGROUND_PROJECT_CPP_CIFAR_DATASET_CIFARTORCHDATASET_H_
#define LIBTORCHPLAYGROUND_PROJECT_CPP_CIFAR_DATASET_CIFARTORCHDATASET_H_

#include <algorithm>
#include <exception>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>


class CIFARTorchDataset : public torch::data::Dataset<CIFARTorchDataset> {
 private:
  // Variables
  struct DataPathStruct {
    std::string targetPath_;
    std::vector<std::string> imagePaths_;
    DataPathStruct(std::string &targetPath, std::vector<std::string> &imagePaths);
  };

  torch::TensorOptions image_tensor_data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  // Functions
  static std::vector<std::string> split_paths(std::string &path);
  static std::string get_target_name(std::string &targetPath);
  static std::vector<CIFARTorchDataset::DataPathStruct> directory_traversal(std::string &path);
  static cv::Mat normalize_cv_image(cv::Mat &img);
  torch::Tensor mat_to_tensor(cv::Mat &img);

 public:
  // Constructor
  explicit CIFARTorchDataset(std::string &path);

  // Destructor
  ~CIFARTorchDataset() override;

  // Variables
  struct LabelEncodingStruct {
    std::string target_;
    int target_id_;
    LabelEncodingStruct(std::string &target, int target_id);
  };
  struct SampleStruct {
    int target_id_;
    std::string imagePath_;
    SampleStruct(int target_id, std::string &imagePath);
  };

  std::vector<LabelEncodingStruct> labelEncoding;
  std::vector<SampleStruct> sampleTable;

  // Functions
  static std::vector<LabelEncodingStruct> generate_label_encoding(std::vector<DataPathStruct> &dataPaths);
  std::vector<SampleStruct> generate_sample_table(std::vector<DataPathStruct> &dataPaths);
  int get_encoded_target(std::string &target_);
  [[maybe_unused]] std::string get_encoded_target(int target_id_);
  torch::Tensor get_image(std::string &image_path);
  torch::data::Example<> get(size_t index) override;
  [[nodiscard]] torch::optional<size_t> size() const override;

  // Operator Overloads
  friend bool operator==(LabelEncodingStruct &labelEncoding_a, LabelEncodingStruct &labelEncoding_b);
  friend bool operator!=(LabelEncodingStruct &labelEncoding_a, LabelEncodingStruct &labelEncoding_b);
  friend bool operator>(LabelEncodingStruct &labelEncoding, std::vector<LabelEncodingStruct> &labelEncodingVector);
};

#endif//LIBTORCHPLAYGROUND_PROJECT_CPP_CIFAR_DATASET_CIFARTORCHDATASET_H_
