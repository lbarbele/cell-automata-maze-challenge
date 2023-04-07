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

  struct Cell {
    static const cell_t dead = 0;
    static const cell_t live = 1;
  };

public:
  const Matrix<cell_t>& config = _config;
  const std::size_t& rows = config.rows;
  const std::size_t& cols = config.cols;

  // copy constructors

  Maze(
    const Maze& other
  ) :
    _config(other.config)
  {}

  // construct from matrix

  Maze(
    const Matrix<cell_t>& m,
    const bool ignore_bad = false
  ) :
    _config(Matrix<cell_t>::full(m.rows, m.cols, Cell::dead))
  {
    // loop over matrix elements to get the initial configuration
    for (std::size_t idx = 0; idx < rows*cols; ++idx) {
      switch (m[idx]) {
      case Cell::dead:
        break;
      case Cell::live:
        set_cell(idx);
        break;
      default:
        if (!ignore_bad) {
          throw std::runtime_error("invalid cell type " + std::to_string(m[idx]));
        }
      }
    }

    // enable the border bit for the border cells
    for (std::size_t i = 0; i < rows; ++i) {
      _config[{i, 0}] |= 2;
      _config[{i, cols-1}] |= 2;
    }

    for (std::size_t j = 1; j < cols - 1; ++j) {
      _config[{0, j}] |= 2;
      _config[{rows-1, j}] |= 2;
    }
  }

  // cell setter

  void set_cell(const std::size_t i, const std::size_t j) {set_cell(i*cols + j);}
  void set_cell(const Position& p) {set_cell(p.x, p.y);}
  
  void
  set_cell(
    const std::size_t idx
  )
  {
    _config[idx] ^= Cell::live;

    const auto i = idx/cols;
    const auto j = idx%cols;

    if (_config[idx] & 2) {
      if (i > 0) {
        _config[idx-cols] += 4;
        if (j < cols-1) {
          _config[idx-cols+1] += 4;
        }
        if (j > 0) {
          _config[idx-cols-1] += 4;
        }
      }

      if (j > 0) {
        _config[idx-1] += 4;
        if (i < rows-1) {
          _config[idx+cols-1] += 4;
        }
      }

      if (i < rows-1) {
        _config[idx+cols] += 4;
        if (j < cols-1) {
          _config[idx+cols+1] += 4;
        }
      }

      if (j < cols-1) {
        _config[idx+1] += 4;
      }
    } else {
      _config[idx-cols-1] += 4;
      _config[idx-cols] += 4;
      _config[idx-cols+1] += 4;
      _config[idx-1] += 4;
      _config[idx+1] += 4;
      _config[idx+cols-1] += 4;
      _config[idx+cols] += 4;
      _config[idx+cols+1] += 4;
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
    _config[idx] ^= Cell::live;

    const auto i = idx/cols;
    const auto j = idx%cols;

    if (_config[idx] & 2) {
      if (i > 0) {
        _config[idx-cols] -= 4;
        if (j < cols-1) {
          _config[idx-cols+1] -= 4;
        }
        if (j > 0) {
          _config[idx-cols-1] -= 4;
        }
      }

      if (j > 0) {
        _config[idx-1] -= 4;
        if (i < rows-1) {
          _config[idx+cols-1] -= 4;
        }
      }

      if (i < rows-1) {
        _config[idx+cols] -= 4;
        if (j < cols-1) {
          _config[idx+cols+1] -= 4;
        }
      }

      if (j < cols-1) {
        _config[idx+1] -= 4;
      }
    } else {
      _config[idx-cols-1] -= 4;
      _config[idx-cols] -= 4;
      _config[idx-cols+1] -= 4;
      _config[idx-1] -= 4;
      _config[idx+1] -= 4;
      _config[idx+cols-1] -= 4;
      _config[idx+cols] -= 4;
      _config[idx+cols+1] -= 4;
    }
  }

  static Maze from_file(std::filesystem::path path, const bool ignore_bad = false)
  {return Maze(Matrix<cell_t>::from_file(path), ignore_bad);}

  Maze&
  evolve(
    const uint generations = 1
  )
  {
    for (uint i = 0; i < generations; ++i) {
      auto m = Matrix<cell_t>(config);

      for (std::size_t idx = 0; idx < rows*cols; ++idx) {
        const auto count = m[idx] / 4;

        if (m[idx] & Cell::live) {
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

  auto operator[](const Position& p) const {return config[p] & Cell::live;}
  auto operator[](const std::size_t idx) const {return config[idx] & Cell::live;}
};

// maze printer
template <class CharT, class Traits, class C>
std::basic_ostream<CharT, Traits>&
operator<<(
  std::basic_ostream<CharT, Traits>& os,
  const Maze<C>& m
) {
  return os << (m.config&1);
}

#endif // _has_maze_hpp_