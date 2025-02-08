#include "rabitQ.hpp"
#include <filesystem>
#include <iostream>
#include <string_view>
#include <unordered_set>
#include <vector>

std::vector<void *> loadFvecs(const std::string &data_path) {
  std::filesystem::path path(data_path);
  if (!std::filesystem::exists(path)) {
    spdlog::error("Data path {} does not exist.", data_path);
    return std::vector<void *>();
  }

  // Load the data from the file
  // The data is stored in the fvecs format
  // The first 4 bytes are the dimension of the vector, remaining bytes are the
  // float values
  auto fio = std::ifstream(data_path, std::ios::binary);

  std::vector<void *> data;
  while (fio) {
    auto dim = readInt32(fio);

    if (dim == 0) {
      break;
    }
    void *vec = new float[dim];
    fio.read(reinterpret_cast<char *>(vec), dim * sizeof(float));
    data.push_back(vec);
  }
  return data;
}

std::vector<std::vector<int>> loadGroundTruth(const std::string &data_path) {
  std::filesystem::path path(data_path);
  if (!std::filesystem::exists(path)) {
    spdlog::error("Data path {} does not exist.", data_path);
    return std::vector<std::vector<int>>();
  }

  // Load the data from the file
  // The data is stored in the fvecs format
  // The first 4 bytes are the dimension of the vector, remaining bytes are the
  // float values
  auto fio = std::ifstream(data_path, std::ios::binary);

  std::vector<std::vector<int>> data;
  while (fio) {
    auto dim = readInt32(fio);

    if (dim == 0) {
      break;
    }
    std::vector<int> vec(dim);
    fio.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(int));
    data.push_back(vec);
  }
  return data;
}

int main(int argc, char *argv[]) {
  std::cout << "Hello, World!" << std::endl;
  constexpr std::string_view data_path = "/home/hayes/projects/rabitQ/data/siftsmall/siftsmall_base.fvecs";
  constexpr std::string_view query_path = "/home/hayes/projects/rabitQ/data/siftsmall/siftsmall_query.fvecs";
  constexpr std::string_view ground_truth_path =
      "/home/hayes/projects/rabitQ/data/siftsmall/siftsmall_groundtruth.ivecs";
  constexpr int dimension = 128;
  rabitQ rabit(data_path.data(), dimension);

  if (rabit.train()) {
    std::cout << "Training successful" << std::endl;
  } else {
    std::cout << "Training failed" << std::endl;
  }

  // Load the query data
  auto query_data = loadFvecs(query_path.data());
  std::vector<std::vector<int>> query_result;
  query_result.resize(query_data.size());
  for (size_t i = 0; i < query_data.size(); ++i) {
    auto result = rabit.search(10, 2, static_cast<float *>(query_data[i]));
    while (!result.empty()) {
      auto [dist, idx] = result.top();
      result.pop();
      query_result[i].push_back(idx);
    }
  }

  // Free the memory
  // for (auto vec : query_data) {
  //   delete[] static_cast<float *>(vec);
  // }

  // Load the ground truth
  auto ground_truth = loadGroundTruth(ground_truth_path.data());

  // convert the ground truth to a set
  std::vector<std::unordered_set<int>> ground_truth_set;
  for (auto &vec : ground_truth) {
    std::unordered_set<int> set(vec.begin(), vec.end());
    ground_truth_set.push_back(set);
  }

  int correct = 0;
  int total_queries = 0;
  for (int i = 0; i < query_result.size(); ++i) {
    for (int idx : query_result[i]) {
      if (ground_truth_set[i].find(idx) != ground_truth_set[i].end()) {
        ++correct;
      }
      ++total_queries;
    }
  }

  double recall_rate = static_cast<double>(correct) / total_queries;
  std::cout << "Recall Rate: " << recall_rate << std::endl;

  return 0;
}