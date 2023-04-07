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
public:
  using val_t = ValT;

private:
  std::vector<ValT> _data;
  std::size_t _rows;
  std::size_t _cols;

  Matrix(const std::size_t m, const std::size_t n, const ValT value)
  : _rows(m),
    _cols(n),
    _data(m*n, value)
  {}

  Matrix(const std::size_t m, const std::size_t n, const std::vector<ValT> data)
  : _rows(m),
    _cols(n),
    _data(data)
  {}

public:

  auto cols() const {return _cols;}
  auto rows() const {return _rows;}
  auto size() const {return _data.size();}

  //
  // element access
  //

  // access by position (tuple)
  auto& operator[](const Position p) {return _data[p.x*cols() + p.y];}
  auto& operator[](const Position p) const {return _data[p.x*cols() + p.y];}

  // access by single index
  auto& operator[](const std::size_t idx) {return _data[idx];}
  auto& operator[](const std::size_t idx) const {return _data[idx];}

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

  //
  // arithmetic operations
  //
  auto operator/(const std::integral auto& b) const
  {
    using RetT = decltype(val_t{}/b);
    auto ret = Matrix<RetT>::full(rows(), cols(), 0);
    for (std::size_t i = 0; i < size(); ++i) ret[i] = _data[i]/b;
    return ret;
  }

  auto operator%(const std::integral auto& b) const
  {
    using RetT = decltype(val_t{}%b);
    auto ret = Matrix<RetT>::full(rows(), cols(), 0);
    for (std::size_t i = 0; i < size(); ++i) ret[i] = _data[i]%b;
    return ret;
  }

  auto operator&(const std::integral auto& b) const
  {
    using RetT = decltype(val_t{}&b);
    auto ret = Matrix<RetT>::full(rows(), cols(), 0);
    for (std::size_t i = 0; i < size(); ++i) ret[i] = _data[i]&b;
    return ret;
  }

  auto operator|(const std::integral auto& b) const
  {
    using RetT = decltype(val_t{}|b);
    auto ret = Matrix<RetT>::full(rows(), cols(), 0);
    for (std::size_t i = 0; i < size(); ++i) ret[i] = _data[i]|b;
    return ret;
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