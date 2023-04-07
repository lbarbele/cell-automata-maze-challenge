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
#include <sstream>
#include <type_traits>
#include <vector>

#include "position.hpp"

namespace utl {

  template <std::integral ValT>
  class matrix {
  public:
    using val_t = ValT;

  private:
    std::vector<ValT> _data;
    std::size_t _rows;
    std::size_t _cols;

    matrix(const std::size_t m, const std::size_t n, const ValT value)
    : _rows(m),
      _cols(n),
      _data(m*n, value)
    {}

    matrix(const std::size_t m, const std::size_t n, const std::vector<ValT>& data)
    : _rows(m),
      _cols(n),
      _data(data)
    {}

  public:

    // capacity

    auto cols() const {return _cols;}
    auto rows() const {return _rows;}
    auto size() const {return _data.size();}
    const auto& data() const {return _data.data();}

    // element access

    auto& operator[](const Position p) {return _data[p.x*cols() + p.y];}
    auto& operator[](const Position p) const {return _data[p.x*cols() + p.y];}

    auto& operator[](const std::size_t idx) {return _data[idx];}
    auto& operator[](const std::size_t idx) const {return _data[idx];}

    // factory functions

    static
    matrix
    full(
      const std::size_t m,
      const std::size_t n,
      const ValT value
    ) {
      return matrix(m, n, value);
    }

    static
    matrix
    from_data(
      const std::size_t m,
      const std::size_t n,
      const std::vector<ValT>& data
    ) {
      if (data.size() != m*n) {
        std::stringstream s;
        s << "data vector of size " << data.size() << " can not be reshaped into " << m + " per " << n + " matrix";
        throw std::runtime_error(s.str());
      }

      return matrix(m, n, data);
    }

    static
    matrix
    from_file(
      std::filesystem::path path
    ) {
      // open file stream from path and check
      std::ifstream file(path);

      if (!file.is_open()) {
        throw std::runtime_error("unable to open file at path " + path.string());
      }

      // start with an empty data vector
      std::vector<ValT> data;

      // process the file line by line and count rows/cols on the fly
      std::string line;
      std::size_t rows = 0;
      std::size_t cols = 0;

      while (std::getline(file, line)) {
        std::stringstream ss(line);
        data.insert(data.end(), std::istream_iterator<ValT>(ss), std::istream_iterator<ValT>());

        ++rows;

        if (!cols) {
          cols = data.size();
        }

        if (data.size() != rows*cols) {
          throw std::runtime_error("invalid matrix file " + path.string());
        }
      }

      return matrix::from_data(rows, cols, data);
    }

    // transformation

    auto
    apply(
      auto op,
      auto... args
    ) -> matrix<std::invoke_result_t<decltype(op), val_t, decltype(args)...>>
      const
    {
      auto m = full(rows(), cols(), 0);
      for (std::size_t idx = 0; idx < size(); ++idx) m[idx] = op(_data[idx], args...);
      return m;
    }

    matrix&
    transform(
      val_t(*op)(const val_t&)
    )
    {
      for (std::size_t idx = 0; idx < size(); ++idx) {
        _data[idx] = op(_data[idx]);
      }
      return *this;
    }


    // arithmetic operations
    
    auto operator/(const std::integral auto& b) const
    {
      using RetT = decltype(val_t{}/b);
      auto ret = matrix<RetT>::full(rows(), cols(), 0);
      for (std::size_t i = 0; i < size(); ++i) ret[i] = _data[i]/b;
      return ret;
    }

    auto operator%(const std::integral auto& b) const
    {
      using RetT = decltype(val_t{}%b);
      auto ret = matrix<RetT>::full(rows(), cols(), 0);
      for (std::size_t i = 0; i < size(); ++i) ret[i] = _data[i]%b;
      return ret;
    }

    auto operator&(const std::integral auto& b) const
    {
      using RetT = decltype(val_t{}&b);
      auto ret = matrix<RetT>::full(rows(), cols(), 0);
      for (std::size_t i = 0; i < size(); ++i) ret[i] = _data[i]&b;
      return ret;
    }

    auto operator|(const std::integral auto& b) const
    {
      using RetT = decltype(val_t{}|b);
      auto ret = matrix<RetT>::full(rows(), cols(), 0);
      for (std::size_t i = 0; i < size(); ++i) ret[i] = _data[i]|b;
      return ret;
    }
  };

  // matrix printer function

  template <class CharT, class Traits, class ValT>
  std::basic_ostream<CharT, Traits>&
  operator<<(
    std::basic_ostream<CharT, Traits>& os,
    const matrix<ValT>& m
  ) {
    for (std::size_t i = 0; i < m.rows(); ++i, os << std::endl)
      for (std::size_t j = 0; j < m.cols(); ++j)
        os << std::setw(2) << m[{i, j}];
    return os;
  }

} // ::utl

#endif // _has_matrix_hpp_