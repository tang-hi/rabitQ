#include "rabitQ.hpp"
#include "Eigen/Dense"
#include "utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/Core/util/Meta.h>
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <queue>
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <thread>
#include <utility>
#include <vector>

bool rabitQ::train() {
  if (!std::filesystem::exists(data_path_)) {
    spdlog::error("Data path does not exist.");
    return false;
  }

  //========= Stage Preprocessing =========
  raw_data_ = loadFvecs(data_path_);
  spdlog::info("Data loaded. Rows: {}, Cols: {}", raw_data_.rows(),
               raw_data_.cols());

  data_size_ = raw_data_.rows();

  auto PT = P_.transpose();

  // 25600 is the default vector number in the cluster
  // cluster_size is the power of 2
  int cluster_size = roundup(std::max(data_size_ / 51200, 1U), 2);

  centroids_ = Matrix::Zero(cluster_size, dimension_);
  indices_.resize(data_size_);
  if (!ivf(cluster_size, raw_data_, centroids_, indices_)) {
    spdlog::error("Training failed. Stage: IVF");
    return false;
  }

  // create the inverted index
  inverted_index_.clear();
  for (int i = 0; i < data_size_; ++i) {
    inverted_index_[indices_[i]].push_back(i);
  }

  transformed_data_ = raw_data_ * PT;
  transformed_centroids_ = centroids_ * PT;

  // calculate the residuals between transformed_data_ and the centroid each row
  // belongs to
  // P.T * (O_r - C) / ||O_r - C||, where O_r is the original row, C is the
  // centroid since ||O_r - C|| will not affect the sign, we can ignore it
  for (int i = 0; i < raw_data_.rows(); ++i) {
    int centroid_index = indices_[i]; // using existing membership in indices_
    transformed_data_.row(i) =
        transformed_data_.row(i) - transformed_centroids_.row(centroid_index);
  }

  // convert the transformed data to binary, eg. if the value is greater than 0
  // the bit is 1 otherwise 0
  // binary_data_ could represent the quantized data
  // eg. P((2 * binary_data - 1) * sqrt(D)) is the quantized data
  binary_data_ = (transformed_data_.array() > 0).cast<uint8_t>();

  // precompute x0 eg. </bar o, o> in the paper
  precomputeX0();

  // pack the binary data to uint64_t format
  packQuantized();

  // do the popcount for the packed data
  precomputePopcount();

  distance_1_.resize(data_size_);
  distance_2_.resize(data_size_);
  error_bound_.resize(data_size_);

  for (int i = 0; i < data_size_; ++i) {
    auto raw_to_centroid = data_dist_to_centroids_[i];
    auto x0 = x0_[i];
    error_bound_[i] = 2 * 1.9 / std::sqrt(static_cast<float>(dimension_ - 1)) *
                      raw_to_centroid * std::sqrt((1 - x0 * x0) / (x0 * x0));
    distance_1_[i] = -2 / norm_factor_ * raw_to_centroid / x0 *
                     (2 * popcount_[i] - dimension_);
    distance_2_[i] = -2 / norm_factor_ * raw_to_centroid / x0;
  }
  return true;
}

void rabitQ::precomputePopcount() {
  popcount_.resize(data_size_);
  for (int i = 0; i < data_size_; ++i) {
    for (int j = 0; j < packed_codec_.cols(); ++j) {
      popcount_[i] += __builtin_popcountll(packed_codec_(i, j));
    }
  }
}

bool rabitQ::save(const std::string &saved_path) {
  std::FILE *ofs = std::fopen(saved_path.c_str(), "wb");
  if (!ofs) {
    spdlog::error("Failed to open file {}", saved_path);
    return false;
  }

  // write the MAGIC number
  std::fwrite(MAGIC.data(), MAGIC.size(), 1, ofs);

  // write the dimension
  write<int>(ofs, dimension_);
  write(ofs, P_);
  write(ofs, raw_data_);
  write(ofs, centroids_);
  write(ofs, data_dist_to_centroids_);
  // write(ofs, transformed_data_);
  // write(ofs, transformed_centroids_);
  // write(ofs, binary_data_);

  write(ofs, data_size_);
  write(ofs, packed_codec_);
  write(ofs, norm_factor_);

  write(ofs, x0_);
  write(ofs, popcount_);

  write(ofs, u_);

  write(ofs, Bq_);
  write(ofs, inverted_index_);
  write(ofs, indices_);
  write(ofs, distance_1_);
  write(ofs, distance_2_);
  write(ofs, error_bound_);
  std::fclose(ofs);
  return true;
}

bool rabitQ::load(const std::string &index_path, bool in_memory) {
  std::FILE *ifs = std::fopen(index_path.c_str(), "rb");
  index_path_ = index_path;
  if (!ifs) {
    spdlog::error("Failed to open file {}", index_path);
    return false;
  }

  std::string magic;
  magic.resize(MAGIC.size());
  std::fread(&magic[0], magic.size(), 1, ifs);
  if (magic != MAGIC) {
    spdlog::error("Invalid magic number");
    return false;
  }

  read(ifs, dimension_);
  read<float>(ifs, P_);
  if (in_memory) {
    read<float>(ifs, raw_data_);
  } else {
    decltype(raw_data_.rows()) rows, cols;
    read(ifs, rows);
    read(ifs, cols);

    // // get the current position
    auto offset = std::ftell(ifs);
    auto data_size = rows * cols * sizeof(float);
    fseek(ifs, offset + data_size, SEEK_SET);
    // read<float>(ifs, raw_data_);
  }
  read<float>(ifs, centroids_);
  read<float>(ifs, data_dist_to_centroids_);
  // read<float>(ifs, transformed_data_);
  // read<float>(ifs, transformed_centroids_);
  // read<uint8_t>(ifs, binary_data_);

  read(ifs, data_size_);
  read<uint64_t>(ifs, packed_codec_);
  read(ifs, norm_factor_);

  read<float>(ifs, x0_);
  read<int>(ifs, popcount_);

  read<float>(ifs, u_);

  read(ifs, Bq_);
  read<int>(ifs, inverted_index_);
  read<int>(ifs, indices_);
  read<float>(ifs, distance_1_);
  read<float>(ifs, distance_2_);
  read<float>(ifs, error_bound_);
  fclose(ifs);
  return true;
}

/**
 * @brief Perform the IVF quantization
 * 1. generate K centroids
 * 2. store the indices of the vectors that belong to each centroid
 * 3. store the vector distances to the centroid it belongs to
 *
 * @param K Number of centroids
 * @param vectors The vectors to quantize
 * @param centroids The centroids
 * @param indices The indices of the centroids
 * @return true if the quantization is successful, false otherwise
 */
bool rabitQ::ivf(int K, rabitQ::Matrix &vectors, rabitQ::Matrix &centroids,
                 std::vector<int> &indices) {
  // random initialization of centroids
  centroids = Matrix::Random(K, dimension_);
  // kmeansPlusPlus(vectors, K, centroids);

  // perform k-means clustering
  int max_iter = 100;
  bool converged = false;
  float threshold = 1e-5;

  Matrix new_centroids = Matrix::Zero(K, dimension_);
  Matrix dists;
  std::vector<uint32_t> assignment_error(K, 0);
  while (!converged && max_iter > 0) {
    std::vector<int> cluster_count(K, 0);
    spdlog::info("Start IVF iteration: {}", 100 - max_iter);
    // assign each vector to the nearest centroid
    Matrix dists = computeDistanceMatrix(vectors, centroids);

    new_centroids.setZero();
    Matrix M = Matrix::Zero(vectors.rows(), K);

    for (int i = 0; i < vectors.rows(); ++i) {
      Eigen::Index min_index;
      dists.row(i).minCoeff(&min_index);
      indices[i] = min_index;
      M(i, indices[i]) = 1;
    }

    for (int i = 0; i < K; i++) {
      Eigen::Index max_index;
      dists.col(i).maxCoeff(&max_index);
      assignment_error[i] = max_index;
    }

    new_centroids = M.transpose() * vectors;

    Eigen::VectorXf counts = M.colwise().sum();

    for (int i = 0; i < K; ++i) {
      if (counts(i) == 0) {
        // reinitialize the centroid
        spdlog::warn("Cluster {} is empty", i);
        new_centroids.row(i) = vectors.row(assignment_error[i]);
      } else {
        new_centroids.row(i) /= counts(i);
      }
    }

    converged = (centroids - new_centroids).norm() < threshold;
    centroids = new_centroids;
    max_iter--;
  }

  spdlog::info("IVF converged: {}, iteration times is {}", converged,
               100 - max_iter);

  // calculate the distance between each vector and the centroid it belongs to
  data_dist_to_centroids_.clear();
  dists = computeDistanceMatrix(vectors, centroids);
  for (int i = 0; i < vectors.rows(); ++i) {
    Eigen::Index min_index;
    auto dist = dists.row(i).minCoeff(&min_index);
    indices[i] = min_index;

    data_dist_to_centroids_.push_back(dist);
  }
  return true;
}

auto rabitQ::search(int K, int nprobe, float *query) -> TopResult {
  nprobe = std::min(nprobe, static_cast<int>(inverted_index_.size()));
  TopResult result;
  auto PT = P_.transpose();

  // convert the query to the transformed domain
  Eigen::Map<Matrix> query_vec(query, 1, dimension_);
  // auto transformed_query = query_vec * PT;

  // find the nearest centroid
  std::vector<std::pair<float, int>> nearest_centroids;
  getNearestCentroids(nprobe, query_vec, nearest_centroids);
  std::vector<TopResult> results;
  std::vector<std::thread> threads;
  results.resize(nprobe);
  threads.reserve(nprobe);
  for (int i = 0; i < nprobe; ++i) {
    threads.emplace_back([&, i] {
      auto [dist, centroid_idx] = nearest_centroids[i];
      auto transformed_query = (query_vec - centroids_.row(centroid_idx)) * PT;
      // nearest_centroids.pop();
      float query_min;
      float width;
      int sum;
      // spdlog::info("scan cluster {} quantizeQuery", centroid_idx);
      auto quantized_query =
          quantizeQuery(transformed_query, centroid_idx, query_min, width, sum);
      // spdlog::info("Quantization completed for centroid {}", centroid_idx);

      ScanContext context;
      context.centroid_idx = centroid_idx;
      context.cluster_dist = dist;
      context.query_min = query_min;
      context.width = width;
      context.sum = sum;
      context.coeff_1 = query_min;
      context.coeff_2 = width;
      context.K = K;

      // TODO(tang-hi): maybe we should avoid the copy here

      context.raw_query = query_vec;
      context.quantized_query = quantized_query;
      scanCluster(context, results[i]);
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
  // merge the results
  for (auto &res : results) {
    while (!res.empty()) {
      auto [dist, idx] = res.top();
      if (result.size() < K) {
        result.push({dist, idx});
      } else {
        if (dist < result.top().first) {
          result.pop();
          result.push({dist, idx});
        }
      }
      res.pop();
    }
  }
  // for (auto [dist, centroid_idx] : nearest_centroids) {
  //   // auto [dist, centroid_idx] = nearest_centroids.top();
  //   // transformed_query is q^{,} in the paper
  //   auto transformed_query = (query_vec - centroids_.row(centroid_idx)) * PT;
  //   // nearest_centroids.pop();
  //   float query_min;
  //   float width;
  //   int sum;
  //   // spdlog::info("scan cluster {} quantizeQuery", centroid_idx);
  //   auto quantized_query =
  //       quantizeQuery(transformed_query, centroid_idx, query_min, width,
  //       sum);
  //   // spdlog::info("Quantization completed for centroid {}", centroid_idx);

  //   ScanContext context;
  //   context.centroid_idx = centroid_idx;
  //   context.cluster_dist = dist;
  //   context.query_min = query_min;
  //   context.width = width;
  //   context.sum = sum;
  //   context.coeff_1 = query_min;
  //   context.coeff_2 = width;
  //   context.K = K;

  //   // TODO(tang-hi): maybe we should avoid the copy here

  //   context.raw_query = query_vec;
  //   context.quantized_query = quantized_query;
  //   scanCluster(context, result);
  // }
  return result;
}

auto rabitQ::quantizeQuery(const Matrix &query, int centroid_idx,
                           float &query_min, float &width, int &sum)
    -> BinaryMatrix {
  query_min = query.minCoeff();
  sum = 0;
  auto query_max = query.maxCoeff();
  width = (query_max - query_min) / ((1 << Bq_) - 1);
  BinaryMatrix quantized_query(1, dimension_);

  for (int i = 0; i < dimension_; ++i) {
    auto quantized_val = (query(0, i) - query_min) / width;
    quantized_val += u_(0, i);
    quantized_query(0, i) = static_cast<uint8_t>(quantized_val);
    // accumulate the quantized value, reuse the sum variable during the query
    sum += quantized_query(0, i);
  }
  return quantized_query;
}

void rabitQ::scanCluster(ScanContext &context, TopResult &result) {
  auto cluster_size = inverted_index_[context.centroid_idx].size();
  constexpr uint32_t rerank_batch = 32;
  std::vector<std::pair<float, int>> rerank_queue;

  // rerange the query to the codec format
  auto dim = context.quantized_query.cols();
  auto query_dim = dim * Bq_ / 64;
  std::vector<uint64_t> query;
  for (int i = 0; i < Bq_; i++) {
    for (int j = 0; j < dim; j += 64) {
      uint64_t val = 0;
      for (int k = 0; k < 64; k++) {
        val |= ((context.quantized_query(0, j + k) & (1 << i)) >> i);
        if (k != 63) {
          val <<= 1;
        }
      }
      query.push_back(val);
    }
  }
  auto calculate_inner_product =
      [&, this](std::vector<uint64_t> &query_codec,
                const PackedMatrix &quantized_data) -> int {
    int sum = 0;
    int sum1 = 0;
    for (int i = 0; i < Bq_; i++) {
      sum1 = 0;
      for (int j = 0; j < dim / 64; j++) {
        sum1 += __builtin_popcountll(quantized_data(0, j) &
                                     query[i * (dim / 64) + j]);
      }
      sum += (sum1 << i);
    }
    return sum;
  };

  for (int i = 0; i < cluster_size; i++) {
    int data_idx = inverted_index_[context.centroid_idx][i];
    float raw_to_centroid = data_dist_to_centroids_[data_idx];
    float raw_query_to_centroid = context.cluster_dist;
    float x0_val = x0_[data_idx];

    float distance_1 = context.coeff_1 * distance_1_[data_idx];
    float distance_2 =
        context.coeff_2 * distance_2_[data_idx] *
        (2 * calculate_inner_product(query, packed_codec_.row(data_idx)) -
         context.sum);

    float distance = raw_to_centroid * raw_to_centroid +
                     raw_query_to_centroid * raw_query_to_centroid +
                     distance_1 + distance_2;

    float error_bound = error_bound_[data_idx] * raw_query_to_centroid;
    estimated_ += 1;

    rerank_queue.push_back({distance - error_bound, data_idx});

    if (rerank_queue.size() == rerank_batch) {
      // may be we could sort the rerank_queue, so we could reduce the disk
      // times
      for (auto &[dist, idx] : rerank_queue) {
        if (result.size() < context.K) {
          // TODO(tang-hi): compute the real distance in the disk, currently is
          // in memory
          auto real_dist = (context.raw_query - getRawData(idx)).squaredNorm();
          result.push({real_dist, idx});
        } else {
          auto max_dist = result.top().first;
          if (dist < max_dist) {
            // TODO(tang-hi): precompute the real distance in the disk,
            // currently is in memory
            auto ground_truth =
                (context.raw_query - getRawData(idx)).squaredNorm();
            if (ground_truth < max_dist) {
              result.push({ground_truth, idx});
              result.pop();
            }
          } else {
            skip_++;
          }
        }
      }
      rerank_queue.clear();
    }
  }

  // // process the remaining elements in the rerank_queue
  for (auto &[dist, idx] : rerank_queue) {
    if (result.size() < context.K) {
      // TODO(tang-hi): compute the real distance in the disk, currently
      // is in memory
      auto real_dist = (context.raw_query - getRawData(idx)).squaredNorm();
      result.push({real_dist, idx});
    } else {
      auto max_dist = result.top().first;
      if (dist < max_dist) {
        // TODO(tang-hi): compute the real distance in the disk, currently
        // is in memory
        auto ground_truth = (context.raw_query - getRawData(idx)).squaredNorm();
        if (ground_truth < max_dist) {
          result.push({ground_truth, idx});
          result.pop();
        }
      } else {
        skip_++;
      }
    }
  }
}

void rabitQ::getNearestCentroids(int nprobe, const Matrix &query,
                                 std::vector<std::pair<float, int>> &result) {
  auto dist = computeDistanceMatrix(query, centroids_);
  for (int i = 0; i < dist.cols(); ++i) {
    result.push_back({dist(0, i), i});
  }
  std::sort(result.begin(), result.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  result.resize(nprobe);
}

void rabitQ::precomputeX0() {

  x0_.resize(data_size_);
  for (int i = 0; i < data_size_; ++i) {

    // ((2 * binary_data - 1) / sqrt(D))
    Matrix converted_data =
        (binary_data_.row(i).cast<float>().array() * 2 - 1) / norm_factor_;

    Matrix residual = transformed_data_.row(i);
    auto residual_norm = residual.norm();
    if (residual_norm == 0) {
      // x0 is the inner product of the vector and quantized vector
      // when the dimension is high, the norm of the vector is 0, but in high
      // dimension the inner product of the vector and the quantized vector is
      // very close to 0.8
      x0_[i] = 0.8;
      continue;
    } else {
      // residual = residual / residual_norm;
      // residual = residual * P_.transpose();
      // x0_[i] = residual_norm
      // calculate the inner product of the vector and the quantized vector
      x0_[i] = (residual * converted_data.transpose())(0, 0) / residual_norm;
    }
  }
}

void rabitQ::packQuantized() {
  // convert binary data to uint64_t codec format
  // already padded to the nearest multiple of 64, it is safe to assume that
  assert(dimension_ % 64 == 0);
  int num_blocks = dimension_ >> 6; // number of uint64_t per row

  // Create a matrix to hold uint64_t packed data with shape (data_size_,
  // dimension_/64)
  packed_codec_ = PackedMatrix::Zero(data_size_, num_blocks);

  for (int i = 0; i < data_size_; ++i) {
    for (int block = 0; block < num_blocks; ++block) {
      uint64_t word = 0;
      // Process 8 columns (each of 8 bits) in the block;
      for (int byte = 0; byte < 8; ++byte) {
        uint8_t byte_val = 0;
        for (int idx = 0; idx < 8; ++idx) {
          int col_idx = block * 64 + byte * 8 + idx;
          byte_val = (byte_val << 1) | (binary_data_(i, col_idx) & 1);
        }
        word = (word << 8) | byte_val;
      }
      packed_codec_(i, block) = word;
    }
  }
}

auto rabitQ::getRawData(int idx) -> Matrix {
  // in memory
  if (raw_data_.rows() > 0) {
    return raw_data_.row(idx);
  }
  auto fd = open(data_path_.c_str(), O_RDONLY | O_DIRECT);
  auto ifs = fdopen(fd, "rb");
  
  if (!ifs) {
    spdlog::error("Failed to open file {}", data_path_);
    return Matrix();
  }
  auto vector_size = dimension_ * sizeof(float) + sizeof(int);
  auto offset = idx * vector_size;
  if (std::fseek(ifs, offset, SEEK_SET) != 0) {
    spdlog::error("Failed to seek to the offset {}", offset);
    return Matrix();
  }

  int32_t dim;
  if (std::fread(&dim, sizeof(int), 1, ifs) != 1) {
    spdlog::error("Failed to read the dimension");
    return Matrix();
  }

  if (dim != dimension_) {
    spdlog::error("Dimension mismatch. Expected: {}, Got: {}", dimension_, dim);
    return Matrix();
  }

  Matrix data = Matrix::Zero(1, dimension_);

  if (std::fread(data.data(), sizeof(float), dimension_, ifs) != dimension_) {
    spdlog::error("Failed to read the data");
    return Matrix();
  }

  std::fclose(ifs);
  return data;
}