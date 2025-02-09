#pragma once
#include "Eigen/Dense"
#include "spdlog/spdlog.h"
#include <filesystem>
#include <fstream>

/**
 * @brief round up a number to the nearest multiple
 *
 * @param numToRound the raw number
 * @param multiple  the multiple to round up to
 * @return int the rounded up number
 */
inline int roundup(int numToRound, int multiple) {
  if (multiple == 0) {
    return numToRound;
  }
  return ((numToRound + multiple - 1) / multiple) * multiple;
}

/**
 * @brief Pad a matrix with zeros (appending columns)
 *
 * @param matrix the matrix to pad
 * @param pad the number of columns to pad
 * @return Eigen::MatrixXf the padded matrix
 */
inline Eigen::MatrixXf padMatrix(const Eigen::MatrixXf &matrix,
                                 int expected_cols) {
  if (matrix.cols() == expected_cols) {
    return matrix;
  }
  if (matrix.cols() > expected_cols) {
    spdlog::error("Matrix has more columns {} than expected {}. truncate it",
                  matrix.cols(), expected_cols);
  }
  Eigen::MatrixXf padded(matrix.rows(), expected_cols);
  padded.block(0, 0, matrix.rows(), matrix.cols()) = matrix;
  return padded;
}

/**
 * @brief Read an integer from a file
 *
 * @param fio the file stream
 * @return int the integer read
 */
inline int readInt32(std::ifstream &fio) {
  int value;
  fio.read(reinterpret_cast<char *>(&value), sizeof(value));
  return value;
}

/**
 * @brief Load the float vectors from the data path
 *
 * @param data_path
 * @return rabitQ::Matrix (N, D) matrix where N is the number of vectors and D
 * is the dimension of the vectors
 */
inline Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
loadFvecs(const std::string &data_path) {
  std::filesystem::path path(data_path);
  if (!std::filesystem::exists(path)) {
    spdlog::error("Data path {} does not exist.", data_path);
    return {};
  }

  // Load the data from the file
  // The data is stored in the fvecs format
  // The first 4 bytes are the dimension of the vector, remaining bytes are the
  // float values
  auto fio = std::ifstream(data_path, std::ios::binary);

  auto file_size = std::filesystem::file_size(path);
  // The first 4 bytes are the dimension of the vector
  auto dim = readInt32(fio);

  if (file_size % (dim * sizeof(float) + sizeof(int)) != 0) {
    spdlog::error("The file size is not a multiple of the vector size");
    return {};
  }
  auto num_vectors = file_size / (dim * sizeof(float) + sizeof(int));

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data(
      num_vectors, dim);

  fio.seekg(0);
  for (int i = 0; i < num_vectors; ++i) {
    // skip the dimension
    fio.seekg(sizeof(int), std::ios::cur);
    fio.read(reinterpret_cast<char *>(data.row(i).data()), dim * sizeof(float));
  }
  return data;
}