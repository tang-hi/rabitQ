#include "rabitQ.hpp"
#include "Eigen/Dense"
#include <filesystem>

bool rabitQ::train() {
  if (!std::filesystem::exists(data_path_)) {
    spdlog::error("Data path does not exist.");
    return false;
  }

  auto fevcs = loadFevcs(data_path_);

  data_size_ = fevcs.rows();

  transformed_data_ = fevcs * P_;

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
