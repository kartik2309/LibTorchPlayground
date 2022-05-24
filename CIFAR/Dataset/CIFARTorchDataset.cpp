//
// Created by Kartik Rajeshwaran on 2022-05-19.
//

#include "CIFARTorchDataset.h"

CIFARTorchDataset::DataPathStruct::DataPathStruct(std::string &targetPath, std::vector<std::string> &imagePaths) {
  targetPath_.assign(targetPath);
  imagePaths_ = imagePaths;
}

CIFARTorchDataset::SampleStruct::SampleStruct(int target_id, std::string &imagePath) {
  target_id_ = target_id;
  imagePath_ = imagePath;
}

CIFARTorchDataset::LabelEncodingStruct::LabelEncodingStruct(std::string &target, int target_id) {
  target_ = target;
  target_id_ = target_id;
}

std::vector<std::string> CIFARTorchDataset::split_paths(std::string &path) {
  std::vector<std::string> splitPath;
  std::string subPath;
  char deliminator = '/';

  for (char &c : path) {
    if (c != deliminator) {
      subPath += c;
      continue;
    }
    splitPath.push_back(subPath);
    subPath.clear();
  }
  splitPath.push_back(subPath);

  return splitPath;
}

std::string CIFARTorchDataset::get_target_name(std::string &targetPath) {
  return split_paths(targetPath).back();
}

std::vector<CIFARTorchDataset::DataPathStruct> CIFARTorchDataset::directory_traversal(std::string &path) {
  std::vector<CIFARTorchDataset::DataPathStruct> dataPaths;
  std::vector<std::string> imagePaths_;
  std::string targetPath_;

  for (const auto &targetDir :
       std::filesystem::recursive_directory_iterator(path)) {
    targetPath_.clear();
    imagePaths_.clear();

    if (targetDir.is_directory()) {

      std::string directorPath = targetDir.path();
      targetPath_.assign(directorPath);

      for (const auto &imageFiles :
           std::filesystem::recursive_directory_iterator(directorPath)) {

        if (imageFiles.is_regular_file()) {
          imagePaths_.push_back(imageFiles.path());
        }
      }
      auto *classData = new DataPathStruct(targetPath_, imagePaths_);
      dataPaths.push_back(*classData);
    }
  }
  return dataPaths;
}

cv::Mat CIFARTorchDataset::normalize_cv_image(cv::Mat &img) {
  img.convertTo(img, CV_32F);
  cv::normalize(img, img, 1, 0, cv::NORM_MINMAX);
  return img;
}
torch::Tensor CIFARTorchDataset::mat_to_tensor(cv::Mat &img) {
  torch::Tensor img_tensor = torch::zeros({img.rows, img.cols, img.channels()}, image_tensor_data_options);
  std::memcpy(img_tensor.data_ptr(), img.data, img_tensor.nbytes());
  img_tensor = img_tensor.permute({2, 0, 1});
  return img_tensor;
}

CIFARTorchDataset::CIFARTorchDataset(std::string &path) {
  std::vector<DataPathStruct> dataPaths = directory_traversal(path);
  labelEncoding = generate_label_encoding(dataPaths);
  sampleTable = generate_sample_table(dataPaths);
}

CIFARTorchDataset::~CIFARTorchDataset() = default;

std::vector<CIFARTorchDataset::LabelEncodingStruct> CIFARTorchDataset::generate_label_encoding(
    std::vector<DataPathStruct> &dataPaths) {

  std::vector<LabelEncodingStruct> labelEncoding;
  int target_id_ = 0;

  for (DataPathStruct &data_path_struct : dataPaths) {
    std::string targetPath_ = data_path_struct.targetPath_;
    std::string target_ = get_target_name(targetPath_);

    LabelEncodingStruct encoding_ = LabelEncodingStruct(target_, target_id_);

    if (!(encoding_ > labelEncoding)) {
      labelEncoding.push_back(encoding_);
    }

    target_id_ += 1;
  }

  return labelEncoding;
}

std::vector<CIFARTorchDataset::SampleStruct> CIFARTorchDataset::generate_sample_table(
    std::vector<CIFARTorchDataset::DataPathStruct> &dataPaths) {
  std::vector<SampleStruct> sampleStructs;
  int target_id;

  for (DataPathStruct &data_path_struct : dataPaths) {
    std::string targetPath = data_path_struct.targetPath_;
    std::string target = get_target_name(targetPath);

    target_id = get_encoded_target(target);

    std::vector<std::string> imagePaths = data_path_struct.imagePaths_;

    for (std::string &imagePath : imagePaths) {
      SampleStruct sampleStruct = SampleStruct(target_id, imagePath);
      sampleStructs.push_back(sampleStruct);
    }
  }

  return sampleStructs;
}

int CIFARTorchDataset::get_encoded_target(std::string &target_) {
  int target_id = -1;

  for (LabelEncodingStruct &encoding_ : labelEncoding) {
    if (encoding_.target_ != target_) {
      continue;
    }
    target_id = encoding_.target_id_;
    break;
  }
  if (target_id == -1) {
    throw std::exception();
  }
  return target_id;
}

std::string CIFARTorchDataset::get_encoded_target(int target_id_) {
  std::string target;

  for (LabelEncodingStruct &encoding_ : labelEncoding) {
    if (encoding_.target_id_ != target_id_) {
      continue;
    }
    target.assign(encoding_.target_);
    break;
  }
  if (target.empty()) {
    throw std::exception();
  }
  return target;
}

torch::Tensor CIFARTorchDataset::get_image(std::string &image_path) {
  cv::Mat img_cv = cv::imread(image_path, cv::IMREAD_COLOR);
  img_cv = normalize_cv_image(img_cv);
  torch::Tensor img_tensor = mat_to_tensor(img_cv);
  return img_tensor;
}

torch::data::Example<> CIFARTorchDataset::get(size_t index) {
  SampleStruct sample = sampleTable.at(index);

  torch::Tensor target_id = torch::full({}, sample.target_id_);
  torch::Tensor image = get_image(sample.imagePath_);

  return {image, target_id};
}

torch::optional<size_t> CIFARTorchDataset::size() const {
  return sampleTable.size();
}

bool operator==(
    CIFARTorchDataset::LabelEncodingStruct &labelEncoding_a,
    CIFARTorchDataset::LabelEncodingStruct &labelEncoding_b) {
  if (
      (labelEncoding_a.target_id_ == labelEncoding_b.target_id_)
      and (labelEncoding_a.target_ == labelEncoding_b.target_))
    return true;
  return false;
}

bool operator!=(
    CIFARTorchDataset::LabelEncodingStruct &labelEncoding_a,
    CIFARTorchDataset::LabelEncodingStruct &labelEncoding_b) {
  if (
      (labelEncoding_a.target_id_ != labelEncoding_b.target_id_)
      and (labelEncoding_a.target_ != labelEncoding_b.target_))
    return true;
  return false;
}

bool operator>(
    CIFARTorchDataset::LabelEncodingStruct &labelEncoding,
    std::vector<CIFARTorchDataset::LabelEncodingStruct> &labelEncodingVector) {

  bool findFlag = false;

  for (CIFARTorchDataset::LabelEncodingStruct &labelEncodingStruct : labelEncodingVector) {
    if (labelEncodingStruct != labelEncoding) {
      continue;
    }
    findFlag = true;
    break;
  }

  return findFlag;
}