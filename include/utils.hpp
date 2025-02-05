#pragma once
#include "Eigen/Dense"
#include "spdlog/spdlog.h"
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