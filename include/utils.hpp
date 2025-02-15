#pragma once
#include "Eigen/Dense"
#include "spdlog/spdlog.h"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <unordered_map>

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

template <typename T> inline void write(std::FILE *ofs, T value) {
  std::fwrite(reinterpret_cast<const char *>(&value), sizeof(value), 1, ofs);
}

template <typename T> inline void read(std::FILE *fio, T &value) {
  std::fread(reinterpret_cast<char *>(&value), sizeof(value), 1, fio);
}

template <typename T>
inline void write(std::FILE *ofs, const std::vector<T> &vec) {
  write(ofs, vec.size());
  std::fwrite(reinterpret_cast<const char *>(vec.data()), sizeof(T), vec.size(),
              ofs);
}

template <typename T> inline void read(std::FILE *fio, std::vector<T> &vec) {
  size_t size;
  std::fread(reinterpret_cast<char *>(&size), sizeof(size), 1, fio);
  vec.resize(size);
  std::fread(reinterpret_cast<char *>(vec.data()), size * sizeof(T), 1, fio);
}

template <typename T>
inline void
write(std::FILE *ofs,
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &matrix) {
  write(ofs, matrix.rows());
  write(ofs, matrix.cols());
  std::fwrite(reinterpret_cast<const char *>(matrix.data()),
              matrix.size() * sizeof(T), 1, ofs);
}

template <typename T>
inline void write(std::FILE *ofs,
                  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor> &matrix) {
  write(ofs, matrix.rows());
  write(ofs, matrix.cols());
  std::fwrite(reinterpret_cast<const char *>(matrix.data()),
              matrix.size() * sizeof(T), 1, ofs);
}

template <typename T>
inline void read(std::FILE *fio,
                 Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &matrix) {
  decltype(matrix.rows()) rows, cols;
  read(fio, rows);
  read(fio, cols);
  matrix.resize(rows, cols);
  std::fread(reinterpret_cast<char *>(matrix.data()), matrix.size() * sizeof(T),
              1, fio);
}

template <typename T>
inline void read(
    std::FILE *fio,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &matrix) {
  decltype(matrix.rows()) rows, cols;
  read(fio, rows);
  read(fio, cols);
  matrix.resize(rows, cols);
  std::fread(reinterpret_cast<char *>(matrix.data()), matrix.size() * sizeof(T), 1, fio);
}

template <typename T>
inline void write(std::FILE *ofs,
                  const std::unordered_map<T, std::vector<T>> &data) {
  write(ofs, data.size());
  for (const auto &pair : data) {
    write(ofs, pair.first);
    write(ofs, pair.second);
  }
}

template <typename T>
inline void read(std::FILE *fio,
                 std::unordered_map<T, std::vector<T>> &data) {
  size_t size;
  read(fio, size);
  data.clear();
  for (size_t i = 0; i < size; ++i) {
    T key;
    std::vector<T> value;
    read(fio, key);
    read(fio, value);
    data[key] = value;
  }
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

/**
 * @brief Compute the distance matrix between vectors and centroids
 *
 * @param vectors the vectors
 * @param centroids the centroids
 * @return Eigen::MatrixXf the distance matrix
 */
inline Eigen::MatrixXf computeDistanceMatrix(const Eigen::MatrixXf &vectors,
                                             const Eigen::MatrixXf &vectors2) {
  Eigen::VectorXf vec_sq = vectors.rowwise().squaredNorm();
  Eigen::VectorXf centroid_sq = vectors2.rowwise().squaredNorm();
  Eigen::MatrixXf dists = -2 * vectors * vectors2.transpose();
  dists = dists.colwise() + vec_sq;
  dists = dists.rowwise() + centroid_sq.transpose();

  // do the sqrt for all the elements in matrix
  return dists.cwiseSqrt();
}

/**
 * @brief Perform the k-means++ initialization
 *
 * @param vectors the vectors to cluster
 * @param K the number of clusters
 * @param centroids the centroids
 */
inline void kmeansPlusPlus(const Eigen::MatrixXf &vectors, int K,
                           Eigen::MatrixXf &centroids) {
  const int n = vectors.rows();
  const int d = vectors.cols();
  assert(K <= n && "K must be less than or equal to the number of vectors");

  // Initialize random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, n - 1);
  std::uniform_real_distribution<float> distrib_real(0.0, 1.0);

  // Select first centroid randomly
  int first_idx = distrib(gen);
  centroids.row(0) = vectors.row(first_idx);

  // Vector to store minimum distances
  Eigen::VectorXf min_distances(n);

  // Select remaining centroids
  for (int k = 1; k < K; k++) {
    // Compute distances to the last added centroid
    auto dists = computeDistanceMatrix(centroids.row(k - 1), vectors);

    // get the minimum distance for each vector
    Eigen::VectorXf min_dists = dists.transpose().rowwise().minCoeff();

    // Update minimum distances
    if (k == 1) {
      min_distances = min_dists;
    } else {
      min_distances = min_distances.cwiseMin(min_dists);
    }

    // Calculate cumulative probabilities
    float sum_distances = min_distances.sum();
    float rand_val = distrib_real(gen) * sum_distances;

    // Select next centroid using weighted probability
    float cumsum = 0.0;
    int next_idx = 0;
    for (int i = 0; i < n; i++) {
      cumsum += min_distances(i);
      if (cumsum >= rand_val) {
        next_idx = i;
        break;
      }
    }

    centroids.row(k) = vectors.row(next_idx);
  }

  spdlog::info("K-means++ initialization completed with {} centroids", K);
}
