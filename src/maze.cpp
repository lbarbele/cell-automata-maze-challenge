#include <exception>
#include <list>

#include "matrix.hpp"
#include "maze.hpp"

Maze::Maze(
  const Maze& other
) :
  _config(other.config)
{}

Maze::Maze(
  const Matrix<cell_t>& m
) :
  _config(Matrix<cell_t>::full(m.rows, m.cols, Cell::dead))
{
  // loop over matrix elements to get the initial configuration
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      // set cell state
      if (m[{i, j}] == Cell::live) {
        set_cell(i, j);
      }

      // enable the border bit for the border cells
      if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
        _config[{i, j}] |= 2;
      }
    }
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
Maze::evolve()
{
  auto m = Matrix<cell_t>(config);

  for (std::size_t idx = 0; idx < rows*cols; ++idx) {
    const auto count = m[idx] / 4;

    if (m[idx] & Cell::live) {
      if ((count < 4 || count > 6)) {
        clear_cell(idx);
      }
    } else {
      if (1 < count && count < 4) {
        set_cell(idx);
      }
    }
  }

  return *this;
}
