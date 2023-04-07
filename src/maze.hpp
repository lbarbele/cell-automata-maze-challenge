#ifndef _has_maze_hpp_
#define _has_maze_hpp_

#include <concepts>
#include <filesystem>

#include "matrix.hpp"
#include "position.hpp"

template <std::unsigned_integral CellT>
class Maze {
public:
  using cell_t = CellT;

private:
  Matrix<cell_t> _config;

  static const cell_t _dead_cell     = 0b000;
  static const cell_t _live_cell     = 0b001;
  static const cell_t _border_bit    = 0b010;
  static const cell_t _count_padding = 0b100;

public:

  // construct from matrix

  Maze(
    const Matrix<cell_t>& m,
    const bool ignore_bad = false
  ) :
    _config(Matrix<cell_t>::full(m.rows(), m.cols(), _dead_cell))
  {
    // loop over matrix elements to get the initial configuration
    for (std::size_t idx = 0; idx < size(); ++idx) {
      switch (m[idx]) {
      case _dead_cell:
        break;
      case _live_cell:
        set_cell(idx);
        break;
      default:
        if (!ignore_bad) {
          throw std::runtime_error("invalid cell type " + std::to_string(m[idx]));
        }
      }
    }

    // enable the border bit for the border cells
    for (std::size_t i = 0; i < rows(); ++i) {
      _config[{i, 0}] |= _border_bit;
      _config[{i, cols()-1}] |= _border_bit;
    }

    for (std::size_t j = 1; j < cols() - 1; ++j) {
      _config[{0, j}] |= _border_bit;
      _config[{rows()-1, j}] |= _border_bit;
    }
  }

  // read maze from file

  static
  Maze
  from_file(
    std::filesystem::path path,
    const bool ignore_bad = false
  )
  {
    return Maze(Matrix<cell_t>::from_file(path), ignore_bad);
  }

  // cell setter

  void set_cell(const std::size_t i, const std::size_t j) {set_cell(i*cols() + j);}
  void set_cell(const Position& p) {set_cell(p.x, p.y);}

  void
  set_cell(
    const std::size_t idx
  )
  {
    if (_config[idx] & _live_cell) {
      return;
    }

    _config[idx] ^= _live_cell;

    if (_config[idx] & _border_bit) {
      const auto i = idx/cols();
      const auto j = idx%cols();

      if (i > 0) {
        _config[idx-cols()] += _count_padding;
        if (j < cols()-1) {
          _config[idx-cols()+1] += _count_padding;
        }
        if (j > 0) {
          _config[idx-cols()-1] += _count_padding;
        }
      }

      if (j > 0) {
        _config[idx-1] += _count_padding;
        if (i < rows()-1) {
          _config[idx+cols()-1] += _count_padding;
        }
      }

      if (i < rows()-1) {
        _config[idx+cols()] += _count_padding;
        if (j < cols()-1) {
          _config[idx+cols()+1] += _count_padding;
        }
      }

      if (j < cols()-1) {
        _config[idx+1] += _count_padding;
      }
    } else {
      _config[idx-cols()-1] += _count_padding;
      _config[idx-cols()] += _count_padding;
      _config[idx-cols()+1] += _count_padding;
      _config[idx-1] += _count_padding;
      _config[idx+1] += _count_padding;
      _config[idx+cols()-1] += _count_padding;
      _config[idx+cols()] += _count_padding;
      _config[idx+cols()+1] += _count_padding;
    }
  }

  // cell clearer

  void clear_cell(const std::size_t i, const std::size_t j) {clear_cell(i*cols + j);}
  void clear_cell(const Position& p) {clear_cell(p.x, p.y);}

  void
  clear_cell(
    const std::size_t idx
  )
  {
    if (~_config[idx] & _live_cell) {
      return;
    }

    _config[idx] ^= _live_cell;

    if (_config[idx] & _border_bit) {
      const auto i = idx/cols();
      const auto j = idx%cols();

      if (i > 0) {
        _config[idx-cols()] -= _count_padding;
        if (j < cols()-1) {
          _config[idx-cols()+1] -= _count_padding;
        }
        if (j > 0) {
          _config[idx-cols()-1] -= _count_padding;
        }
      }

      if (j > 0) {
        _config[idx-1] -= 4;
        if (i < rows()-1) {
          _config[idx+cols()-1] -= _count_padding;
        }
      }

      if (i < rows()-1) {
        _config[idx+cols()] -= _count_padding;
        if (j < cols()-1) {
          _config[idx+cols()+1] -= _count_padding;
        }
      }

      if (j < cols()-1) {
        _config[idx+1] -= _count_padding;
      }
    } else {
      _config[idx-cols()-1] -= _count_padding;
      _config[idx-cols()] -= _count_padding;
      _config[idx-cols()+1] -= _count_padding;
      _config[idx-1] -= _count_padding;
      _config[idx+1] -= _count_padding;
      _config[idx+cols()-1] -= _count_padding;
      _config[idx+cols()] -= _count_padding;
      _config[idx+cols()+1] -= _count_padding;
    }
  }

  // evolve maze to the next generation

  Maze&
  evolve(
    const uint generations = 1
  )
  {
    for (uint i = 0; i < generations; ++i) {
      auto m = Matrix<cell_t>(config());

      for (std::size_t idx = 0; idx < size(); ++idx) {
        const auto& state = m[idx];
        const auto count = state / _count_padding;

        if (state & _live_cell) {
          if (count <= 3 || count >= 6) {
            clear_cell(idx);
          }
        } else {
          if (1 < count && count < 5) {
            set_cell(idx);
          }
        }
      }
    }

    return *this;
  }

  // functions to retrieve the state of a particular cell

  auto operator[](const Position& p) const {return config[p] & _live_cell;}
  auto operator[](const std::size_t idx) const {return config[idx] & _live_cell;}

  // utility

  const auto& config() const {return _config;}
  auto cols() const {return config().cols();}
  auto rows() const {return config().rows();}
  auto size() const {return config().size();}

};

// maze printer
template <class CharT, class Traits, class C>
std::basic_ostream<CharT, Traits>&
operator<<(
  std::basic_ostream<CharT, Traits>& os,
  const Maze<C>& m
) {
  return os << (m.config()&1);
}

#endif // _has_maze_hpp_