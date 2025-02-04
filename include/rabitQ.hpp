#pragma once
#include "Eigen/Dense"
#include "spdlog/spdlog.h"
#include "utils.hpp"
#include <cstdint>
#include <string>

class rabitQ {
public:
  using Matrix = Eigen::MatrixXf;

  rabitQ(std::string data_path, int dimension)
      : data_path_(std::move(data_path)), dimension_(dimension) {

    if (dimension_ % 64 != 0) {
      spdlog::warn(
          "The dimension {} is not a multiple of 64, it will be rounded "
          "up to the nearest multiple of 64. eg {}",
          dimension_, roundup(dimension_, 64));
      dimension_ = roundup(dimension_, 64);
    }

    P_ = Matrix::Random(dimension_, dimension_);
    Eigen::HouseholderQR<Matrix> qr(P_);
    P_ = qr.householderQ();
  }

  bool train();

  bool save(const std::string &saved_path);

  bool load(const std::string &index_path);

  ~rabitQ() = default;

private:
  Matrix loadFevcs(const std::string &data_path);

private:
  // data path is the path to the vector, file format should be fvecs
  std::string data_path_;

  // dimension_ may be larger than the actual dimension of the data
  // because we will pad the data to the nearest multiple of 64.
  // so the quantized data will be a multiple of 64(uint64_t).
  int dimension_;

  // P is a random orthogonal matrix.
  Matrix P_;

  Matrix transformed_data_;

  uint32_t data_size_;
};