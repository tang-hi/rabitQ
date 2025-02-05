#include "rabitQ.hpp"
#include "Eigen/Dense"
#include "utils.hpp"
#include <cstdint>
#include <filesystem>

bool rabitQ::train() {
  if (!std::filesystem::exists(data_path_)) {
    spdlog::error("Data path does not exist.");
    return false;
  }

  //========= Stage Preprocessing =========
  auto fevcs = loadFevcs(data_path_);

  data_size_ = fevcs.rows();

  auto PT = P_.transpose();

  // 256 is the default vector number in the cluster
  // cluster_size is the power of 2
  int cluster_size = roundup(std::min(data_size_ / 256, 1U), 2);

  transformed_data_ = fevcs * P_;

  Matrix centroids = Matrix::Zero(cluster_size, dimension_);
  std::vector<int> indices(cluster_size);
  if (!ivf(cluster_size, transformed_data_, centroids, indices)) {
    spdlog::error("Training failed. Stage: IVF");
    return false;
  }

  transformed_centroids_ = centroids * P_;

  // calculate the residuals between transformed_data_ and the centroid each row
  // belongs to
  for (int i = 0; i < transformed_data_.rows(); ++i) {
    int centroid_index = indices[i]; // using existing membership in indices
    transformed_data_.row(i) =
        transformed_data_.row(i) - transformed_centroids_.row(centroid_index);
  }

  // convert the transformed data to binary, eg. if the value is greater than 0
  // the bit is 1 otherwise 0
  binary_data_ = (transformed_data_.array() > 0).cast<uint8_t>();

  precomputeX0();

  packQuantized();

  //========= Stage Indexing =========

  return true;
}

// TODO: Implement this function
bool rabitQ::save(const std::string &saved_path) { return false; }

// TODO: Implement this function
bool rabitQ::load(const std::string &index_path) { return false; }

// TODO: Implement this function
/**
 * @brief Load the float vectors from the data path
 *
 * @param data_path
 * @return rabitQ::Matrix (N, D) matrix where N is the number of vectors and D
 * is the dimension of the vectors
 */
rabitQ::Matrix rabitQ::loadFevcs(const std::string &data_path) {
  return Matrix();
}

// TODO: Implement this function
/**
 * @brief Perform the IVF quantization
 *
 * @param K Number of centroids
 * @param vectors The vectors to quantize
 * @param centroids The centroids
 * @param indices The indices of the centroids
 * @return true if the quantization is successful, false otherwise
 */
bool rabitQ::ivf(int K, rabitQ::Matrix &vectors, rabitQ::Matrix &centroids,
                 std::vector<int> &indices) {
  return false;
}

void rabitQ::precomputeX0() {
  // precompute x0
  // x0 is the inner product of the vector and quantized vector
  float norm_factor = std::sqrt(static_cast<float>(dimension_));
  x0_.resize(data_size_, 1);
  for (int i = 0; i < data_size_; ++i) {
    float row_sum = ((transformed_data_.row(i).array() *
                      ((binary_data_.row(i).cast<float>().array() * 2) - 1)) /
                     norm_factor)
                        .sum();
    float row_norm = transformed_data_.row(i).norm();
    if (row_norm == 0) {
      // when the dimension is high, the norm of the vector is 0, but in high
      // dimension the inner product of the vector and the quantized vector is
      // very close to 0.8
      x0_(i, 0) = 0.8;
    } else {
      x0_(i, 0) = row_sum / row_norm;
    }
  }
}

void rabitQ::packQuantized() {
  // convert binary data to uint64_t codec format

  int num_blocks = dimension_ >> 6; // number of uint64_t per row

  // Create a matrix to hold uint64_t packed data with shape (data_size_,
  // dimension_/64)
  packed_codec_(data_size_, num_blocks);
  for (int i = 0; i < data_size_; ++i) {
    for (int block = 0; block < num_blocks; ++block) {
      uint64_t word = 0;
      // Process 8 rows (each of 8 bits) in the block; note: reversing the
      // order of rows
      for (int j = 0; j < 8; ++j) {
        uint8_t byte_val = 0;
        int row_idx = 7 - j; // reverse row order within the block
        for (int k = 0; k < 8; ++k) {
          int col_idx = block * 64 + row_idx * 8 + k;
          byte_val = (byte_val << 1) | (binary_data_(i, col_idx) & 1);
        }
        word = (word << 8) | byte_val;
      }
      packed_codec_(i, block) = word;
    }
  }
}