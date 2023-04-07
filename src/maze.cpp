#include <exception>
#include <vector>

#include "matrix.hpp"
#include "maze.hpp"

Maze::Maze(
  const Maze& other
) :
  _config(other.config)
{}

Maze::Maze(
  const Matrix<cell_t>& m,
  const bool ignore_bad
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

void
Maze::set_cell(
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

void
Maze::clear_cell(
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

Maze&
Maze::evolve(const uint generations)
{
  for (uint i = 0; i < generations; ++i) {
    auto m = Matrix<cell_t>(config);

    for (std::size_t idx = 0; idx < rows*cols; ++idx) {
      const auto count = m[idx] / 4;

      if (m[idx] & Cell::live) {
        if (count < 4 || count > 6) {
          clear_cell(idx);
        }
      } else {
        if (1 < count && count < 4) {
          set_cell(idx);
        }
      }
    }
  }

  return *this;
}
