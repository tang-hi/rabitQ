#include "rabitQ.hpp"
#include <filesystem>
#include <iostream>
#include <spdlog/spdlog.h>
#include <string_view>
#include <unordered_set>
#include <vector>

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
  constexpr std::string_view data_path =
      "/home/hayes/projects/rabitQ/data/gist/gist_base.fvecs";
  constexpr std::string_view query_path =
      "/home/hayes/projects/rabitQ/data/gist/gist_query.fvecs";
  constexpr std::string_view ground_truth_path =
      "/home/hayes/projects/rabitQ/data/gist/gist_groundtruth.ivecs";
  constexpr int dimension = 960;
  auto ground_truth = loadGroundTruth(ground_truth_path.data());
  auto K = 100;
  // convert the ground truth to a set
  std::vector<std::unordered_set<int>> ground_truth_set;
  for (auto &vec : ground_truth) {
    std::unordered_set<int> set(vec.begin(), vec.begin() + K);
    ground_truth_set.push_back(set);
  }

  rabitQ rabit(data_path.data(), dimension);

  if (rabit.train()) {
    spdlog::info("Training succeeded");
  } else {
    spdlog::error("Training failed");
  }

  auto data = loadFvecs(data_path.data());

  // Load the query data
  rabitQ::TopResult result;

  auto query_data = loadFvecs(query_path.data());

  int correct = 0;
  int total_queries = 0;

  for (size_t i = 0; i < query_data.rows(); ++i) {
    spdlog::info("Processing query {}", i);
    auto result = rabit.search(K, 4, query_data.row(i).data());
    while (!result.empty()) {
      auto [dist, idx] = result.top();
      if (ground_truth_set[i].find(idx) != ground_truth_set[i].end()) {
        correct++;
      }
      total_queries++;
      result.pop();
    }
    double recall_rate = static_cast<double>(correct) / total_queries;
    spdlog::info("Query: {} ,Recall Rate: {}", i, recall_rate);
    // spdlog::info("Wrong estimation: {}",
    //              float(rabit.wrong_estimate()) / rabit.estimated());
  }

  // Load the ground truth

  double recall_rate = static_cast<double>(correct) / total_queries;
  std::cout << "Recall Rate: " << recall_rate << std::endl;

  spdlog::info("Skipped: {}", rabit.skipped());

  return 0;
}
