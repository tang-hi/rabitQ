#include "rabitQ.hpp"
#include "Eigen/Dense"
#include "utils.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <queue>
#include <spdlog/spdlog.h>
#include <utility>
#include <vector>

bool rabitQ::train() {
  if (!std::filesystem::exists(data_path_)) {
    spdlog::error("Data path does not exist.");
    return false;
  }

  //========= Stage Preprocessing =========
  raw_data_ = loadFevcs(data_path_);

  data_size_ = raw_data_.rows();

  auto PT = P_.transpose();

  // 256 is the default vector number in the cluster
  // cluster_size is the power of 2
  int cluster_size = roundup(std::max(data_size_ / 5000, 1U), 2);

  centroids_ = Matrix::Zero(cluster_size, dimension_);
  std::vector<int> indices(data_size_);
  if (!ivf(cluster_size, raw_data_, centroids_, indices)) {
    spdlog::error("Training failed. Stage: IVF");
    return false;
  }

  // create the inverted index
  inverted_index_.clear();
  for (int i = 0; i < data_size_; ++i) {
    inverted_index_[indices[i]].push_back(i);
  }

  transformed_data_ = raw_data_ * PT;
  transformed_centroids_ = centroids_ * PT;

  // calculate the residuals between transformed_data_ and the centroid each row
  // belongs to
  // P.T * (O_r - C) / ||O_r - C||, where O_r is the original row, C is the
  // centroid since ||O_r - C|| will not affect the sign, we can ignore it
  for (int i = 0; i < transformed_data_.rows(); ++i) {
    int centroid_index = indices[i]; // using existing membership in indices
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

// TODO: Implement this function
bool rabitQ::save(const std::string &saved_path) { return false; }

// TODO: Implement this function
bool rabitQ::load(const std::string &index_path) { return false; }

/**
 * @brief Load the float vectors from the data path
 *
 * @param data_path
 * @return rabitQ::Matrix (N, D) matrix where N is the number of vectors and D
 * is the dimension of the vectors
 */
rabitQ::Matrix rabitQ::loadFevcs(const std::string &data_path) {
  std::filesystem::path path(data_path);
  if (!std::filesystem::exists(path)) {
    spdlog::error("Data path {} does not exist.", data_path);
    return Matrix();
  }

  // Load the data from the file
  // The data is stored in the fvecs format
  // The first 4 bytes are the dimension of the vector, remaining bytes are the
  // float values
  auto fio = std::ifstream(data_path, std::ios::binary);

  Matrix data;
  while (fio) {
    auto dim = readInt32(fio);

    if (dim == 0) {
      break;
    }
    Eigen::MatrixXf vec(1, dim);
    fio.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));
    data.conservativeResize(data.rows() + 1, dim);
    data.row(data.rows() - 1) = vec;
  }
  return data;
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

  // perform k-means clustering
  int max_iter = 100;
  bool converged = false;

  while (!converged && max_iter > 0) {
    std::vector<int> cluster_counts(K, 0);
    // assign each vector to the nearest centroid
    Matrix new_centroids = Matrix::Zero(K, dimension_);
    for (int i = 0; i < vectors.rows(); ++i) {
      float min_dist = std::numeric_limits<float>::max();
      int min_idx = -1;
      for (int j = 0; j < K; ++j) {
        float dist = (vectors.row(i) - centroids.row(j)).norm();
        if (dist < min_dist) {
          min_dist = dist;
          min_idx = j;
        }
      }
      indices[i] = min_idx;
      cluster_counts[min_idx]++;
      new_centroids.row(min_idx) += vectors.row(i);
    }

    // update the centroids
    for (int cluster_id = 0; cluster_id < K; ++cluster_id) {
      if (cluster_counts[cluster_id] == 0) {
        // if no vector is assigned to the centroid, reinitialize it
        spdlog::warn(
            "Centroid {} has no vectors assigned to it. Reinitializing",
            cluster_id);
        centroids.row(cluster_id) = Matrix::Random(1, dimension_);
      } else {
        new_centroids.row(cluster_id) /= cluster_counts[cluster_id];
        if ((new_centroids.row(cluster_id) - centroids.row(cluster_id)).norm() <
            1e-6) {
          converged = true;
        } else {
          converged = false;
        }
        // update the centroid
        centroids.row(cluster_id) = new_centroids.row(cluster_id);
      }
    }
    max_iter--;
  }

  spdlog::info("IVF converged: {}, iteration times is {}", converged,
               100 - max_iter);

  // calculate the distance between each vector and the centroid it belongs to
  data_dist_to_centroids_.clear();
  for (int i = 0; i < vectors.rows(); ++i) {
    int centroid_index = indices[i];
    data_dist_to_centroids_.push_back(
        (vectors.row(i) - centroids.row(centroid_index)).norm());
  }
  return true;
}

auto rabitQ::search(int K, int nprobe, float *query) -> TopResult {
  TopResult result;
  auto PT = P_.transpose();

  // convert the query to the transformed domain
  Eigen::Map<Matrix> query_vec(query, 1, dimension_);
  // auto transformed_query = query_vec * PT;

  // find the nearest centroid
  TopResult nearest_centroids;
  getNearestCentroids(nprobe, query_vec, nearest_centroids);

  while (!nearest_centroids.empty()) {
    auto residual_query =
        query_vec - centroids_.row(nearest_centroids.top().second);

    // transformed_query is q^{,} int the paper
    auto transformed_query = residual_query * PT;
    auto [dist, centroid_idx] = nearest_centroids.top();
    nearest_centroids.pop();
    int query_min;
    float width;
    int sum;
    auto quantized_query =
        quantizeQuery(transformed_query, centroid_idx, query_min, width, sum);

    ScanContext context;
    context.centroid_idx = centroid_idx;
    context.cluster_dist = dist;
    context.query_min = query_min;
    context.width = width;
    context.sum = sum;
    context.coeff_1 = -2 * query_min / norm_factor_;
    context.coeff_2 = -2 * width / norm_factor_;
    context.error_bound_coeff =
        2 * (1.9 / std::sqrt(static_cast<float>(dimension_ - 1)));

    // TODO(tang-hi): maybe we should avoid the copy here
    context.raw_query = query_vec;
    context.quantized_query = quantized_query;

    scanCluster(context, result);
  }
  return result;
}

auto rabitQ::quantizeQuery(const Matrix &query, int centroid_idx,
                           int &query_min, float &width, int &sum)
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

  // TODO(tang-hi): need to optimize the inner product calculation
  auto calculate_inner_product =
      [](const BinaryMatrix &quantized_query,
         const PackedMatrix &quantized_data) -> uint64_t {
    uint64_t inner_product = 0;
    for (int i = 0; i < quantized_data.cols(); ++i) {
      for (int j = 0; j < 64; j++) {
        inner_product += ((quantized_data(0, i)) & (1 << (63 - j)) ? 1 : 0) *
                         ((quantized_query(0, i * 64 + j)));
      }
    }
    return inner_product;
  };

  for (int i = 0; i < cluster_size; i++) {
    int data_idx = inverted_index_[context.centroid_idx][i];
    float raw_to_centroid = data_dist_to_centroids_[data_idx];
    float raw_query_to_centroid = context.cluster_dist;
    float x0_val = x0_[data_idx];

    float distance_1 = context.coeff_1 * raw_query_to_centroid / x0_val *
                       (2 * popcount_[data_idx] - dimension_);

    float distance_2 =
        context.coeff_2 * raw_to_centroid / x0_val *
        (2 * calculate_inner_product(context.quantized_query,
                                     packed_codec_.row(data_idx)) -
         context.sum);

    float distance = raw_to_centroid * raw_to_centroid +
                     raw_query_to_centroid * raw_query_to_centroid +
                     distance_1 + distance_2;

    float error_bound = context.error_bound_coeff * raw_to_centroid *
                        raw_query_to_centroid *
                        std::sqrt((1 - x0_val * x0_val) / (x0_val * x0_val));

    rerank_queue.push_back({distance - error_bound, data_idx});

    if (rerank_queue.size() == rerank_batch) {
      // may be we could sort the rerank_queue, so we could reduce the disk
      // times
      for (auto &[dist, idx] : rerank_queue) {
        if (result.size() < context.K) {
          // TODO(tang-hi): precompute the real distance in the disk, currently
          // is in memory
          auto real_dist = (context.raw_query - raw_data_.row(idx)).norm();
          result.push({real_dist, idx});
        } else {
          auto max_dist = result.top().first;
          if (dist < max_dist) {
            // TODO(tang-hi): precompute the real distance in the disk,
            // currently is in memory
            auto ground_truth = (context.raw_query - raw_data_.row(idx)).norm();
            if (ground_truth < max_dist) {
              result.pop();
              result.push({ground_truth, idx});
            }
          }
        }
      }
      rerank_queue.clear();
    }
  }

  // process the remaining elements in the rerank_queue
  for (auto &[dist, idx] : rerank_queue) {
    if (result.size() < context.K) {
      // TODO(tang-hi): precompute the real distance in the disk, currently
      // is in memory
      auto real_dist = (context.raw_query - raw_data_.row(idx)).norm();
      result.push({real_dist, idx});
    } else {
      auto max_dist = result.top().first;
      if (dist < max_dist) {
        // TODO(tang-hi): precompute the real distance in the disk, currently
        // is in memory
        auto ground_truth = (context.raw_query - raw_data_.row(idx)).norm();
        if (ground_truth < max_dist) {
          result.pop();
          result.push({ground_truth, idx});
        }
      }
    }
  }
  return;
}

void rabitQ::getNearestCentroids(int nprobe, const Matrix &query,
                                 TopResult &result) {
  for (int i = 0; i < centroids_.rows(); ++i) {
    float dist = (query - centroids_.row(i)).norm();
    if (result.size() < nprobe) {
      result.push({dist, i});
    } else {
      auto max_dist = result.top().first;
      if (dist < max_dist) {
        result.pop();
        result.push({dist, i});
      }
    }
  }
}

void rabitQ::precomputeX0() {

  x0_.resize(data_size_);
  for (int i = 0; i < data_size_; ++i) {

    // (2 * binary_data - 1) * sqrt(D) dot P
    auto converted_data =
        (binary_data_.row(i).cast<float>().array() * 2 - 1) * norm_factor_;
    auto quantized_data = converted_data.matrix() * P_;

    auto residual = raw_data_.row(i) - centroids_.row(i);
    auto residual_norm = residual.norm();
    if (residual_norm == 0) {
      // x0 is the inner product of the vector and quantized vector
      // when the dimension is high, the norm of the vector is 0, but in high
      // dimension the inner product of the vector and the quantized vector is
      // very close to 0.8
      x0_[i] = 0.8;
      continue;
    } else {
      // calculate the inner product of the vector and the quantized vector
      x0_[i] = (quantized_data * residual.transpose())(0, 0) / residual_norm;
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