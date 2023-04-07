#ifndef _has_matrix_hpp_
#define _has_matrix_hpp_

#include <concepts>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>

#include "position.hpp"

template <std::integral ValT>
class Matrix {
private:
  std::vector<ValT> data;
  std::size_t _rows;
  std::size_t _cols;

  Matrix(const std::size_t m, const std::size_t n, const ValT value)
  : _rows(m),
    _cols(n),
    data(m*n, value)
  {}

  Matrix(const std::size_t m, const std::size_t n, const std::vector<ValT> data)
  : _rows(m),
    _cols(n),
    data(data)
  {}

public:
  using value_type = ValT;
  const std::size_t& rows = _rows;
  const std::size_t& cols = _cols;

  //
  // copy constructor
  //
  template <std::convertible_to<ValT> U>
  Matrix(
    const Matrix<U>& other
  ) :
    data(other.data.begin(), other.data.end()),
    _rows(other.rows),
    _cols(other.cols)
  {}

  //
  // element access
  //

  // access by position (tuple)
  auto& operator[](const Position p) {return data[p.x*cols + p.y];}
  auto& operator[](const Position p) const {return data[p.x*cols + p.y];}

  // access by single index
  auto& operator[](const std::size_t idx) {return data[idx];}
  auto& operator[](const std::size_t idx) const {return data[idx];}

  //
  // functions to create a matrix
  //

  // create a matrix of size m x n filled with given value
  static Matrix full(const std::size_t m, const std::size_t n, const ValT value) {
    return Matrix(m, n, value);
  }

  // create a matrix from a vector of data
  static Matrix from_data(const std::size_t m, const std::size_t n, const std::vector<ValT> data) {
    if (data.size() != m*n) {
      throw std::runtime_error("data vector of size " + std::to_string(data.size()) + " can not be reshaped into " + std::to_string(m) + " per " + std::to_string(n) + " matrix");
    }

    return Matrix(m, n, data);
  }

  // create a matrix whose elements are listed into a file, one line per row
  static Matrix from_file(std::filesystem::path path) {
    // open file stream from path and check
    std::ifstream file(path);

    if (!file.is_open()) {
      throw std::runtime_error("unable to open file at path " + path.string());
    }

    // start with an empty data vector
    std::vector<ValT> data;

    // process the file line by line and count rows on the fly
    std::string line;
    std::size_t rows = 0;

    while (std::getline(file, line)) {
      std::stringstream ss(line);
      data.insert(data.end(), std::istream_iterator<ValT>(ss), std::istream_iterator<ValT>());
      ++rows;
    }

    // create the matrix object
    return Matrix::from_data(rows, data.size()/rows, data);
  }
};

// matrix printer function
template <class CharT, class Traits, class ValT>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const Matrix<ValT>& m) {
  for (std::size_t i = 0; i < m.rows*m.cols; ++i) {
    os << std::setw(2) << m[i];
    if ((i+1)%m.cols == 0)
      os << std::endl;
  }
  return os;
}


#endif