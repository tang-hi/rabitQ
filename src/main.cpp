#include "rabitQ.hpp"
#include <filesystem>
#include <iostream>
#include <spdlog/spdlog.h>
#include <string>
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
  if (argc != 5 && argc != 7) {
    spdlog::error("Usage: {}  build <data_path> <index_path> <dim>", argv[0]);
    spdlog::error("Usage: {}  search <index_path> <data_path> <query_path> "
                  "<ground_truth_path> <dim>",
                  argv[0]);
    return -1;
  }

  std::string cmd = argv[1];
  if (cmd != "build" && cmd != "search") {
    spdlog::error("Invalid command: {}", cmd);
    return -1;
  }
  if (cmd == "build") {
    std::string data_path = argv[2];
    std::string index_path = argv[3];
    int dimension = std::stoi(argv[4]);
    if (!std::filesystem::exists(data_path)) {
      spdlog::error("Data path {} does not exist.", data_path);
      return -1;
    }

    rabitQ rabit(data_path, dimension);
    if (!rabit.train()) {
      spdlog::error("Training failed");
      return -1;
    }
    rabit.save(index_path);
    spdlog::info("Index saved to {}", index_path);
    return 0;
  } else if (cmd == "search") {
    std::string index_path = argv[2];
    std::string data_path = argv[3];
    std::string query_path = argv[4];
    std::string ground_truth_path = argv[5];
    int dimension = std::stoi(argv[6]);
    if (!std::filesystem::exists(index_path)) {
      spdlog::error("Index path {} does not exist.", index_path);
      return -1;
    }
    if (!std::filesystem::exists(query_path)) {
      spdlog::error("Query path {} does not exist.", query_path);
      return -1;
    }
    if (!std::filesystem::exists(ground_truth_path)) {
      spdlog::error("Ground truth path {} does not exist.", ground_truth_path);
      return -1;
    }
    auto ground_truth = loadGroundTruth(ground_truth_path);
    auto K = 100;
    // convert the ground truth to a set
    std::vector<std::unordered_set<int>> ground_truth_set;
    for (auto &vec : ground_truth) {
      std::unordered_set<int> set(vec.begin(), vec.begin() + K);
      ground_truth_set.push_back(set);
    }
    rabitQ rabitQ(data_path, dimension);
    if (!rabitQ.load(index_path, false)) {
      spdlog::error("Failed to load the index");
      return -1;
    }
    rabitQ::TopResult result;

    auto query_data = loadFvecs(query_path.data());

    int correct = 0;
    int total_queries = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < query_data.rows(); ++i) {
      auto result = rabitQ.search(K, 4, query_data.row(i).data());
      while (!result.empty()) {
        auto [dist, idx] = result.top();
        if (ground_truth_set[i].find(idx) != ground_truth_set[i].end()) {
          correct++;
        }
        total_queries++;
        result.pop();
      }
      double recall_rate = static_cast<double>(correct) / total_queries;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    spdlog::info("Elapsed time: {}", elapsed_seconds.count());
    spdlog::info("Average time: {} ms",
                 elapsed_seconds.count() / query_data.rows() * 1000);

    // Load the ground truth

    double recall_rate = static_cast<double>(correct) / total_queries;
    std::cout << "Recall Rate: " << recall_rate << std::endl;
    spdlog::info("Skipped: {}", rabitQ.skipped());

  } else {
    spdlog::error("Invalid command");
    return -1;
  }
  return 0;
}
