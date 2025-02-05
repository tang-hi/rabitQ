#include "rabitQ.hpp"
#include "Eigen/Dense"
#include "utils.hpp"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <queue>
#include <spdlog/spdlog.h>

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
    spdlog::error("Data path does not exist.");
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

  // random initialization of centroids
  centroids = Matrix::Random(K, dimension_);

  // perform k-means clustering
  int max_iter = 100;
  bool converged = false;

  // TODO(tang-hi): add the corresponding data to the inverted list
  while (!converged && max_iter > 0) {
    // assign each vector to the nearest centroid
    std::vector<int> counts(K, 0);
    Matrix new_centroids = Matrix::Zero(K, dimension_);
    for (int i = 0; i < vectors.rows(); ++i) {
      float min_dist = std::numeric_limits<float>::max();
      int min_idx = -1;
      for (int j = 0; j < K; ++j) {
        float dist = (vectors.row(i) - centroids.row(j)).squaredNorm();
        if (dist < min_dist) {
          min_dist = dist;
          min_idx = j;
        }
      }
      indices[i] = min_idx;
      new_centroids.row(min_idx) += vectors.row(i);
      counts[min_idx]++;
    }

    // update the centroids
    for (int i = 0; i < K; ++i) {
      if (counts[i] == 0) {
        // if no vector is assigned to the centroid, reinitialize it
        spdlog::warn(
            "Centroid {} has no vectors assigned to it. Reinitializing", i);
        centroids.row(i) = Matrix::Random(1, dimension_);
      } else {
        new_centroids.row(i) /= counts[i];
        if ((new_centroids.row(i) - centroids.row(i)).norm() < 1e-6) {
          converged = true;
        } else {
          converged = false;
        }
        centroids.row(i) = new_centroids.row(i);
      }
    }
    max_iter--;
  }

  spdlog::info("IVF converged: {}, iteration times is {}", converged,
               100 - max_iter);
  return true;
}

auto rabitQ::search(int K, int nprobe, float *query) -> TopResult {
  TopResult result;
  auto PT = P_.transpose();

  // convert the query to the transformed domain
  Eigen::Map<Matrix> query_vec(query, 1, dimension_);
  auto transformed_query = query_vec * PT;

  // find the nearest centroid
  TopResult nearest_centroids;
  getNearestCentroids(nprobe, transformed_query, nearest_centroids);

  while (!nearest_centroids.empty()) {
    auto [negative_dist, centroid_idx] = nearest_centroids.top();
    nearest_centroids.pop();
    int residual_min;
    float width;
    int sum;
    auto quantized_query = quantizeQuery(transformed_query, centroid_idx,
                                         residual_min, width, sum);
  }
}

auto rabitQ::quantizeQuery(const Matrix &query, int centroid_idx,
                           int &residual_min, float &width, int &sum)
    -> Matrix {
  auto centroid = transformed_centroids_.row(centroid_idx);
  auto residual_query = query.row(0) - centroid;
  auto residual_query_min = residual_query.minCoeff();
  auto residual_query_max = residual_query.maxCoeff();
  width = (residual_query_max - residual_query_min) / ((1 << Bq_) - 1);
  residual_min = residual_query_min;
  BinaryMatrix quantized_query(1, dimension_);
  for (int i = 0; i < dimension_; ++i) {
    quantized_query(0, i) = static_cast<uint8_t>(
        (residual_query(0, i) - residual_query_min) / width);

    sum += quantized_query(0, i);
  }
  return quantized_query;
}

void rabitQ::scanCluster(const Matrix &query, float cluster_dist,
                         int cluster_id, TopResult &result) {
  constexpr int size = 32;
  auto cluster_size = inverted_index_[cluster_id].size();
  uint32_t i = 0;
  for (; i < cluster_size; i += size) {
    for (int j = 0; j < size; ++j) {
      float tmp_dist =
          data_dist_to_centroids_[inverted_index_[cluster_id][i + j]] +
          cluster_dist + /* ptr_fac->factor_ppc */ residual_min +
          /* (distance between quant_query and packed_codec * 2 - sumq ) */ 0 *
              /* (ptr_fac->factor_ip)*/ width;

      float error_bound = cluster_dist * 1 /* ptr_fac->error*/;
    }
  }

  for (; i < cluster_size; ++i) {
    float tmp_dist =
        data_dist_to_centroids_[inverted_index_[cluster_id][i]] + cluster_dist +
        /* ptr_fac->factor_ppc */ residual_min +
        /* (distance between quant_query and packed_codec * 2 - sumq ) */ 0 *
            /* (ptr_fac->factor_ip)*/ width;

    float error_bound = cluster_dist * 1 /* ptr_fac->error*/;
    if (result.size() < K) {
      auto ground_truth =
          query[0] *
          transformed_data_.row(inverted_index_[cluster_id][i]).transpose();
      result.push({-ground_truth(0, 0), inverted_index_[cluster_id][i]});
    } else {
      auto max_dist = -1 * result.top().second;
      if ((tmp_dist - error_bound) < max_dist) {
        auto ground_truth =
            query[0] *
            transformed_data_.row(inverted_index_[cluster_id][i]).transpose();
        if (ground_truth(0, 0) < max_dist) {
          result.pop();
          result.push({-ground_truth(0, 0), inverted_index_[cluster_id][i]});
        }
      }
    }
  }
}

void rabitQ::getNearestCentroids(int nprobe, const Matrix &transformed_query,
                                 TopResult &result) {
  for (int i = 0; i < transformed_centroids_.rows(); ++i) {
    float dist = (transformed_query - transformed_centroids_.row(i)).norm();
    if (result.size() < nprobe) {
      result.push({-dist, i});
    } else {
      auto max_dist = -1 * result.top().second;
      if (dist < max_dist) {
        result.pop();
        result.push({-dist, i});
      }
    }
  }
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